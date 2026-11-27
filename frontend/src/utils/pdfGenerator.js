/**
 * Clinical Consultation Report PDF Generator
 * 
 * Generates a clean, professional, publication-grade clinical document
 * conforming to PDF-1.4 standards without external library dependencies.
 */

/**
 * Unescapes HTML entities (hex, decimal, named) to prevent corrupted entity display.
 */
export function decodeHtmlEntities(str) {
  if (!str || typeof str !== 'string') return str;
  let res = str;
  for (let i = 0; i < 5; i++) {
    const prev = res;
    res = res
      .replace(/&#x([0-9a-fA-F]+);?/gi, (_, hex) => String.fromCharCode(parseInt(hex, 16)))
      .replace(/&#([0-9]+);?/g, (_, dec) => String.fromCharCode(parseInt(dec, 10)))
      .replace(/&quot;/gi, '"')
      .replace(/&apos;/gi, "'")
      .replace(/&amp;/gi, '&')
      .replace(/&lt;/gi, '<')
      .replace(/&gt;/gi, '>');
    if (res === prev) break;
  }
  if (typeof document !== 'undefined') {
    try {
      const txt = document.createElement('textarea');
      txt.innerHTML = res;
      res = txt.value;
    } catch (_) {}
  }
  return res;
}

/**
 * Escapes characters for PDF literal strings (parentheses, backslashes)
 * and maps special Unicode characters to clean ASCII representations.
 */
function sanitizePdfText(str) {
  if (!str) return '';
  const decoded = decodeHtmlEntities(String(str));
  return decoded
    .replace(/\\/g, '\\\\')
    .replace(/\(/g, '\\(')
    .replace(/\)/g, '\\)')
    .replace(/[–—]/g, '-')
    .replace(/[""]/g, '"')
    .replace(/['']/g, "'")
    .replace(/•/g, '-')
    .replace(/°/g, ' deg ')
    .replace(/·/g, '-')
    .replace(/🚨/g, '[!]')
    .replace(/✓/g, '[Y]')
    .replace(/[^\x20-\x7E\xA0-\xFF]/g, ' ')
    .trim();
}

/**
 * Wraps text into lines that fit within a maximum width (in PDF points).
 * Standard Helvetica average character width ~0.52 * fontSize.
 */
function wrapText(text, maxWidth, fontSize) {
  if (!text) return [];
  const charWidth = fontSize * 0.52;
  const maxChars = Math.max(10, Math.floor(maxWidth / charWidth));
  const words = String(text).split(/\s+/);
  const lines = [];
  let currentLine = '';

  for (const word of words) {
    if (!word) continue;
    const testLine = currentLine ? `${currentLine} ${word}` : word;
    if (testLine.length <= maxChars) {
      currentLine = testLine;
    } else {
      if (currentLine) lines.push(currentLine);
      if (word.length > maxChars) {
        // Break long words
        let remaining = word;
        while (remaining.length > maxChars) {
          lines.push(remaining.slice(0, maxChars));
          remaining = remaining.slice(maxChars);
        }
        currentLine = remaining;
      } else {
        currentLine = word;
      }
    }
  }
  if (currentLine) lines.push(currentLine);
  return lines;
}

/**
 * Parse raw report text into structured clinical sections.
 */
export function parseClinicalReport(reportText) {
  if (!reportText) return null;

  const raw = decodeHtmlEntities(String(reportText));
  const lines = raw.split('\n');

  // Section mapping
  const sections = {
    chiefComplaint: {},
    hpi: '',
    symptoms: { timelines: [] },
    vitals: {},
    medicalHistory: {},
    medications: {},
    allergies: {},
    lifestyle: {},
    demographics: {},
    redFlags: null,
    recommendations: [],
    clinicalSummary: '',
    generatedAt: '',
    isEmergency: false,
    rawLines: lines
  };

  let currentSection = null;
  const sectionContent = {};

  const headerAliases = [
    { key: 'chiefComplaint', match: ['CHIEF COMPLAINT', 'Chief Complaint'] },
    { key: 'hpi', match: ['HISTORY OF PRESENT ILLNESS (HPI)', 'HISTORY OF PRESENT ILLNESS', 'History of Present Illness'] },
    { key: 'symptoms', match: ['SYMPTOMS', 'Symptoms', 'Symptoms & Timeline'] },
    { key: 'vitals', match: ['VITAL SIGNS', 'Vital Signs', 'Vitals'] },
    { key: 'medicalHistory', match: ['MEDICAL HISTORY', 'Medical History'] },
    { key: 'medications', match: ['MEDICATIONS', 'Medications', 'Current Medications'] },
    { key: 'allergies', match: ['ALLERGIES', 'Allergies'] },
    { key: 'lifestyle', match: ['LIFESTYLE', 'Lifestyle', 'Social History'] },
    { key: 'demographics', match: ['DEMOGRAPHICS', 'Demographics', 'Patient Profile'] },
    { key: 'redFlags', match: ['RED FLAGS', 'Red Flags', 'RED FLAGS / EMERGENCY ASSESSMENT', 'Emergency & Red Flag Assessment'] },
    { key: 'recommendations', match: ['RECOMMENDATIONS & SUGGESTED NEXT STEPS', 'RECOMMENDATIONS', 'Recommendations', 'CLINICAL RECOMMENDATIONS', 'Suggested Clinical Next Steps', 'Suggested Clinical Next Steps:'] },
    { key: 'clinicalSummary', match: ['CLINICAL SUMMARY', 'Clinical Summary', 'Summary for Physician'] }
  ];

  for (const rawLine of lines) {
    const trimmed = rawLine.trim();
    if (!trimmed || trimmed.startsWith('===') || trimmed.startsWith('---')) continue;

    // Check generated timestamp
    if (trimmed.toLowerCase().startsWith('generated:')) {
      sections.generatedAt = trimmed.split(':', 2)[1]?.trim() || '';
      continue;
    }

    // Check emergency warning
    if (trimmed.includes('EMERGENCY ESCALATION REQUIRED') || trimmed.includes('EMERGENCY WARNING')) {
      sections.isEmergency = true;
    }

    // Check section headers
    let matchedKey = null;
    const cleanHeader = trimmed.replace(/[*#_]/g, '').trim();

    for (const item of headerAliases) {
      for (const m of item.match) {
        if (cleanHeader.toUpperCase() === m.toUpperCase() || cleanHeader.toUpperCase().startsWith(`${m.toUpperCase()}:`)) {
          matchedKey = item.key;
          break;
        }
      }
      if (matchedKey) break;
    }

    if (matchedKey) {
      currentSection = matchedKey;
      if (!sectionContent[currentSection]) {
        sectionContent[currentSection] = [];
      }
      // If header had inline content (e.g. "Chief Complaint: Headache")
      if (cleanHeader.includes(':')) {
        const afterColon = cleanHeader.substring(cleanHeader.indexOf(':') + 1).trim();
        if (afterColon) sectionContent[currentSection].push(afterColon);
      }
    } else if (currentSection) {
      sectionContent[currentSection].push(trimmed);
    }
  }

  // Parse Chief Complaint
  if (sectionContent.chiefComplaint) {
    for (const l of sectionContent.chiefComplaint) {
      const cleaned = l.replace(/[*_]/g, '').trim();
      if (cleaned.toLowerCase().startsWith('reason:')) {
        sections.chiefComplaint.reason = cleaned.substring(7).trim();
      } else if (cleaned.toLowerCase().startsWith('duration:')) {
        sections.chiefComplaint.duration = cleaned.substring(9).trim();
      } else if (cleaned.toLowerCase().startsWith('status:')) {
        sections.chiefComplaint.status = cleaned.substring(7).trim();
      } else if (!sections.chiefComplaint.reason) {
        sections.chiefComplaint.reason = cleaned;
      }
    }
  }

  // Parse HPI
  if (sectionContent.hpi) {
    sections.hpi = sectionContent.hpi.join(' ').replace(/[*_]/g, '').trim();
  }

  // Parse Symptoms
  if (sectionContent.symptoms) {
    let inTimelines = false;
    for (const l of sectionContent.symptoms) {
      const cleaned = l.replace(/[*_]/g, '').trim();
      const lower = cleaned.toLowerCase();
      if (lower.startsWith('symptom timelines')) {
        inTimelines = true;
        continue;
      }
      if (inTimelines && cleaned.startsWith('-')) {
        const item = cleaned.substring(1).trim();
        if (item) sections.symptoms.timelines.push(item);
        continue;
      }
      inTimelines = false;
      if (lower.startsWith('reported:')) {
        sections.symptoms.reported = cleaned.substring(9).trim();
      } else if (lower.startsWith('location:')) {
        sections.symptoms.location = cleaned.substring(9).trim();
      } else if (lower.startsWith('severity:')) {
        sections.symptoms.severity = cleaned.substring(9).trim();
      } else if (lower.startsWith('character:')) {
        sections.symptoms.character = cleaned.substring(10).trim();
      } else if (lower.startsWith('onset:')) {
        sections.symptoms.onset = cleaned.substring(6).trim();
      } else if (lower.startsWith('timing:')) {
        sections.symptoms.timing = cleaned.substring(7).trim();
      } else if (lower.startsWith('frequency:')) {
        sections.symptoms.frequency = cleaned.substring(10).trim();
      } else if (lower.startsWith('progression:')) {
        sections.symptoms.progression = cleaned.substring(12).trim();
      } else if (lower.includes('pertinent negatives') || lower.includes('explicitly denied:')) {
        const val = cleaned.substring(cleaned.indexOf(':') + 1).trim();
        if (val) {
          sections.symptoms.negatives = sections.symptoms.negatives
            ? `${sections.symptoms.negatives}, ${val}`
            : val;
        }
      } else if (cleaned.startsWith('-') && !lower.startsWith('---')) {
        const item = cleaned.substring(1).trim();
        if (item) {
          sections.symptoms.negatives = sections.symptoms.negatives
            ? `${sections.symptoms.negatives}, ${item}`
            : item;
        }
      }
    }
  }

  // Parse Vitals
  if (sectionContent.vitals) {
    for (const l of sectionContent.vitals) {
      const cleaned = l.replace(/[*_]/g, '').trim();
      const lower = cleaned.toLowerCase();
      if (lower.startsWith('blood pressure:')) {
        sections.vitals.bloodPressure = cleaned.substring(15).trim();
      } else if (lower.startsWith('heart rate:')) {
        sections.vitals.heartRate = cleaned.substring(11).trim();
      } else if (lower.startsWith('temperature:')) {
        sections.vitals.temperature = cleaned.substring(12).trim();
      } else if (lower.startsWith('respiratory rate:')) {
        sections.vitals.respiratoryRate = cleaned.substring(17).trim();
      } else if (lower.startsWith('oxygen saturation:')) {
        sections.vitals.oxygenSaturation = cleaned.substring(18).trim();
      } else if (lower.startsWith('blood glucose:')) {
        sections.vitals.bloodGlucose = cleaned.substring(14).trim();
      }
    }
  }

  // Parse Medical History
  if (sectionContent.medicalHistory) {
    for (const l of sectionContent.medicalHistory) {
      const cleaned = l.replace(/[*_]/g, '').trim();
      const lower = cleaned.toLowerCase();
      if (lower.startsWith('conditions:')) {
        sections.medicalHistory.conditions = cleaned.substring(11).trim();
      } else if (lower.startsWith('surgeries:')) {
        sections.medicalHistory.surgeries = cleaned.substring(10).trim();
      } else if (lower.startsWith('hospitalizations:')) {
        sections.medicalHistory.hospitalizations = cleaned.substring(17).trim();
      } else if (lower.startsWith('family history:')) {
        sections.medicalHistory.familyHistory = cleaned.substring(15).trim();
      }
    }
  }

  // Parse Medications
  if (sectionContent.medications) {
    for (const l of sectionContent.medications) {
      const cleaned = l.replace(/[*_]/g, '').trim();
      if (cleaned.toLowerCase().startsWith('current:')) {
        sections.medications.current = cleaned.substring(8).trim();
      } else if (!sections.medications.current && cleaned.length > 2) {
        sections.medications.current = cleaned;
      }
    }
  }

  // Parse Allergies
  if (sectionContent.allergies) {
    for (const l of sectionContent.allergies) {
      const cleaned = l.replace(/[*_]/g, '').trim();
      const lower = cleaned.toLowerCase();
      if (lower.startsWith('drug:')) {
        sections.allergies.drug = cleaned.substring(5).trim();
      } else if (lower.startsWith('food:')) {
        sections.allergies.food = cleaned.substring(5).trim();
      } else if (lower.startsWith('environmental:')) {
        sections.allergies.environmental = cleaned.substring(14).trim();
      }
    }
  }

  // Parse Lifestyle
  if (sectionContent.lifestyle) {
    for (const l of sectionContent.lifestyle) {
      const cleaned = l.replace(/[*_]/g, '').trim();
      const lower = cleaned.toLowerCase();
      if (lower.startsWith('smoking:')) {
        sections.lifestyle.smoking = cleaned.substring(8).trim();
      } else if (lower.startsWith('alcohol:')) {
        sections.lifestyle.alcohol = cleaned.substring(8).trim();
      } else if (lower.startsWith('exercise:')) {
        sections.lifestyle.exercise = cleaned.substring(9).trim();
      } else if (lower.startsWith('diet:')) {
        sections.lifestyle.diet = cleaned.substring(5).trim();
      } else if (lower.startsWith('recent travel:')) {
        sections.lifestyle.recentTravel = cleaned.substring(14).trim();
      } else if (lower.startsWith('sick contacts:')) {
        sections.lifestyle.sickContacts = cleaned.substring(14).trim();
      }
    }
  }

  // Parse Demographics
  if (sectionContent.demographics) {
    for (const l of sectionContent.demographics) {
      const cleaned = l.replace(/[*_]/g, '').trim();
      const lower = cleaned.toLowerCase();
      if (lower.startsWith('age:')) {
        sections.demographics.age = cleaned.substring(4).trim();
      } else if (lower.startsWith('gender:')) {
        sections.demographics.gender = cleaned.substring(7).trim();
      }
    }
  }

  // Parse Red Flags
  if (sectionContent.redFlags) {
    const flagsText = sectionContent.redFlags.join(' ');
    sections.redFlags = {
      detected: !flagsText.toLowerCase().includes('no red flags detected'),
      raw: sectionContent.redFlags.join('\n')
    };
  }

  // Parse Recommendations
  if (sectionContent.recommendations) {
    sections.recommendations = sectionContent.recommendations
      .map((l) => l.replace(/^[-*•]\s*/, '').trim())
      .filter((l) => l.length > 0 && !l.startsWith('===') && !l.startsWith('---'));
  }

  // Parse Clinical Summary
  if (sectionContent.clinicalSummary) {
    // Filter out standard disclaimer tails and Human Validation footer
    const summaryLines = sectionContent.clinicalSummary.filter((l) => {
      const low = l.toLowerCase();
      return (
        !l.startsWith('---') &&
        !low.includes('this report contains patient-reported') &&
        !low.includes('end of report') &&
        !low.includes('human validation required') &&
        !low.includes("if you're a healthcare professional") &&
        !low.includes("if you're a patient:")
      );
    });
    sections.clinicalSummary = summaryLines.join('\n').replace(/[*_]/g, '').trim();
  }

  // Check emergency keywords in raw text
  const lowerRaw = raw.toLowerCase();
  if (lowerRaw.includes('emergency escalation') || lowerRaw.includes('immediate medical attention') || lowerRaw.includes('[!] emergency')) {
    sections.isEmergency = true;
  }

  // Deep decode all string values across all parsed fields
  function deepDecode(val) {
    if (typeof val === 'string') return decodeHtmlEntities(val);
    if (Array.isArray(val)) return val.map(deepDecode);
    if (val && typeof val === 'object') {
      const out = {};
      for (const [k, v] of Object.entries(val)) {
        out[k] = deepDecode(v);
      }
      return out;
    }
    return val;
  }

  return deepDecode(sections);
}

/**
 * Generate a complete PDF binary buffer for a clinical report.
 * 
 * @param {string} reportText - The raw report text from clinical intake
 * @param {Object} metadata - Optional metadata (sessionId, isConfirmed, confirmationTimestamp)
 * @returns {Uint8Array} Binary PDF data
 */
export function buildClinicalReportPdf(reportText, metadata = {}) {
  const parsed = parseClinicalReport(reportText);
  if (!parsed) {
    throw new Error('Cannot generate PDF: Report data is empty or invalid.');
  }

  // Page geometry (A4 standard: 595.28 x 841.89 points)
  const pageWidth = 595.28;
  const pageHeight = 841.89;
  const margin = 36.0;
  const usableWidth = pageWidth - 2 * margin; // 523.28 pt
  const colWidth = (usableWidth - 14) / 2; // ~254.6 pt each

  const dateStr = parsed.generatedAt
    ? parsed.generatedAt.slice(0, 10)
    : new Date().toISOString().slice(0, 10);
  const sessionRef = metadata.sessionId ? metadata.sessionId.slice(-8).toUpperCase() : 'INTAKE';

  // Multi-page builder
  const pages = [];
  let currentCommands = [];
  let currentY = pageHeight - 45; // Start near top

  function startPage() {
    currentCommands = [];
    currentY = pageHeight - 45;
  }

  function finishPage() {
    pages.push(currentCommands.join('\n'));
  }

  function checkPageBreak(requiredHeight) {
    if (currentY - requiredHeight < 65) {
      finishPage();
      startPage();
      // Draw continuation header on subsequent pages
      drawHeaderBanner(true);
    }
  }

  // Drawing helpers
  function drawRect(x, y, w, h, fillRgb, strokeRgb = null, lineWidth = 0.5) {
    currentCommands.push('q');
    if (fillRgb) {
      currentCommands.push(`${fillRgb[0]} ${fillRgb[1]} ${fillRgb[2]} rg`);
    }
    if (strokeRgb) {
      currentCommands.push(`${strokeRgb[0]} ${strokeRgb[1]} ${strokeRgb[2]} RG`);
      currentCommands.push(`${lineWidth} w`);
    }
    const op = fillRgb && strokeRgb ? 'B' : fillRgb ? 'f' : 'S';
    currentCommands.push(`${x.toFixed(2)} ${y.toFixed(2)} ${w.toFixed(2)} ${h.toFixed(2)} re ${op}`);
    currentCommands.push('Q');
  }

  function drawLine(x1, y1, x2, y2, strokeRgb, lineWidth = 0.5) {
    currentCommands.push('q');
    currentCommands.push(`${strokeRgb[0]} ${strokeRgb[1]} ${strokeRgb[2]} RG`);
    currentCommands.push(`${lineWidth} w`);
    currentCommands.push(`${x1.toFixed(2)} ${y1.toFixed(2)} m ${x2.toFixed(2)} ${y2.toFixed(2)} l S`);
    currentCommands.push('Q');
  }

  function drawText(text, x, y, font = '/F2', fontSize = 9, colorRgb = [0.1, 0.15, 0.2]) {
    const sanitized = sanitizePdfText(text);
    if (!sanitized) return;
    currentCommands.push('BT');
    currentCommands.push(`${colorRgb[0]} ${colorRgb[1]} ${colorRgb[2]} rg`);
    currentCommands.push(`${font} ${fontSize} Tf`);
    currentCommands.push(`${x.toFixed(2)} ${y.toFixed(2)} Td`);
    currentCommands.push(`(${sanitized}) Tj`);
    currentCommands.push('ET');
  }

  function drawHeaderBanner(isContinuation = false) {
    // Top banner background
    drawRect(margin, currentY - 32, usableWidth, 36, [0.96, 0.98, 1.0], [0.85, 0.90, 0.96], 1);
    // Navy accent left stripe
    drawRect(margin, currentY - 32, 4, 36, [0.08, 0.25, 0.48]);

    drawText('Clinical Consultation Report', margin + 12, currentY - 14, '/F1', 14, [0.06, 0.18, 0.35]);
    drawText(
      isContinuation
        ? '(Continued from Page 1)'
        : 'AI-Assisted Pre-Consultation Summary | Outpatient Clinical Intake',
      margin + 12,
      currentY - 26,
      '/F2',
      8,
      [0.35, 0.42, 0.52]
    );

    // Right-aligned metadata
    const metaX = margin + usableWidth - 130;
    drawText(`Date: ${dateStr}`, metaX, currentY - 14, '/F1', 8, [0.15, 0.20, 0.30]);
    drawText(`Ref: ${sessionRef} | Pre-Consult`, metaX, currentY - 24, '/F2', 7.5, [0.40, 0.45, 0.55]);

    currentY -= 42;
  }

  // --- BEGIN DOCUMENT GENERATION ---
  startPage();
  drawHeaderBanner(false);

  // Patient Info & Verification Strip
  const infoHeight = 22;
  drawRect(margin, currentY - infoHeight, usableWidth, infoHeight, [0.98, 0.99, 1.0], [0.90, 0.92, 0.95], 0.5);

  const ageText = parsed.demographics.age ? `Age: ${parsed.demographics.age}` : 'Age: Adult';
  const genderText = parsed.demographics.gender ? `Gender: ${parsed.demographics.gender}` : '';
  const demoText = [ageText, genderText].filter(Boolean).join('  |  ');
  const verifyText = metadata.isConfirmed ? 'Verified by Patient [Y]' : 'Patient-Reported (Unverified)';
  const verifyColor = metadata.isConfirmed ? [0.05, 0.50, 0.25] : [0.45, 0.50, 0.55];

  drawText(`Demographics: ${demoText}`, margin + 8, currentY - 14, '/F1', 8, [0.20, 0.25, 0.35]);
  drawText(`Intake: Completed`, margin + 200, currentY - 14, '/F2', 8, [0.30, 0.35, 0.45]);
  drawText(verifyText, margin + usableWidth - 140, currentY - 14, '/F1', 8, verifyColor);

  currentY -= (infoHeight + 8);

  // Emergency Alert Box (if applicable)
  if (parsed.isEmergency) {
    const alertHeight = 26;
    drawRect(margin, currentY - alertHeight, usableWidth, alertHeight, [1.0, 0.94, 0.94], [0.90, 0.25, 0.25], 1);
    drawText('[!] EMERGENCY WARNING: ACUTE CLINICAL EVALUATION RECOMMENDED', margin + 10, currentY - 12, '/F1', 8.5, [0.80, 0.10, 0.10]);
    drawText('Patient report contains critical indicators requiring immediate medical assessment.', margin + 10, currentY - 22, '/F2', 7.5, [0.60, 0.10, 0.10]);
    currentY -= (alertHeight + 8);
  }

  // Chief Complaint Bar
  const ccReason = parsed.chiefComplaint.reason || 'Symptom evaluation (pre-consultation)';
  const ccDuration = parsed.chiefComplaint.duration ? ` (Duration: ${parsed.chiefComplaint.duration})` : '';
  const ccFullText = `${ccReason}${ccDuration}`;
  const ccLines = wrapText(ccFullText, usableWidth - 110, 8.5);
  const ccBoxHeight = Math.max(22, 12 + ccLines.length * 10);

  drawRect(margin, currentY - ccBoxHeight, usableWidth, ccBoxHeight, [0.95, 0.97, 0.99], [0.85, 0.88, 0.92], 0.5);
  drawText('CHIEF COMPLAINT:', margin + 8, currentY - 13, '/F1', 8.5, [0.08, 0.25, 0.48]);
  ccLines.forEach((l, i) => {
    drawText(l, margin + 108, currentY - 13 - i * 10, '/F2', 8.5, [0.10, 0.15, 0.20]);
  });
  currentY -= (ccBoxHeight + 8);

  // --- TWO-COLUMN CLINICAL FINDINGS SECTION ---
  const colYStart = currentY;
  let leftY = colYStart;
  let rightY = colYStart;

  // Helper for drawing boxed field cards
  function drawSectionCard(x, startY, width, title, fields, iconTag = '') {
    let y = startY;
    // Section Header
    drawRect(x, y - 14, width, 14, [0.92, 0.95, 0.98], [0.80, 0.86, 0.92], 0.5);
    drawText(title, x + 6, y - 10, '/F1', 7.8, [0.08, 0.25, 0.48]);
    if (iconTag) {
      drawText(iconTag, x + width - 60, y - 10, '/F3', 6.5, [0.45, 0.50, 0.58]);
    }
    y -= 14;

    const bodyStartY = y;

    // First pass: measure height
    let totalContentHeight = 6;
    const preparedFields = [];
    for (const f of fields) {
      if (!f || !f.value) continue;
      const label = f.label ? `${f.label}: ` : '';
      const wrapped = wrapText(f.value, width - 14 - label.length * 4.2, 7.5);
      preparedFields.push({ f, label, wrapped });
      totalContentHeight += (Math.max(1, wrapped.length) * 9.5 + 2);
    }

    const cardHeight = Math.max(20, totalContentHeight + 4);
    // Draw card background
    drawRect(x, bodyStartY - cardHeight, width, cardHeight, [1.0, 1.0, 1.0], [0.86, 0.89, 0.93], 0.5);

    // Draw fields once
    let fieldY = y - 9;
    for (const { f, label, wrapped } of preparedFields) {
      drawText(label, x + 6, fieldY, '/F1', 7.5, [0.25, 0.30, 0.40]);
      const valueX = x + 6 + (label ? label.length * 4.2 : 0);
      wrapped.forEach((wLine, wIdx) => {
        const valColor = f.isMuted ? [0.55, 0.60, 0.65] : [0.10, 0.15, 0.25];
        drawText(wLine, wIdx === 0 ? valueX : x + 14, fieldY - wIdx * 9.5, '/F2', 7.5, valColor);
      });
      fieldY -= (Math.max(1, wrapped.length) * 9.5 + 2);
    }

    return bodyStartY - cardHeight - 6;
  }

  // LEFT COLUMN: Symptoms & Timeline + Vital Signs
  const leftX = margin;
  const symptomFields = [
    { label: 'Reported', value: parsed.symptoms.reported || 'None reported' },
    { label: 'Location', value: parsed.symptoms.location || 'Not specified', isMuted: !parsed.symptoms.location },
    { label: 'Severity', value: parsed.symptoms.severity || 'Not specified', isMuted: !parsed.symptoms.severity },
    { label: 'Character', value: parsed.symptoms.character || 'Not specified', isMuted: !parsed.symptoms.character },
    { label: 'Onset', value: parsed.symptoms.onset || 'Not specified', isMuted: !parsed.symptoms.onset },
    { label: 'Progression', value: parsed.symptoms.progression || 'Not specified', isMuted: !parsed.symptoms.progression },
    ...(parsed.symptoms.timelines && parsed.symptoms.timelines.length > 0
      ? [{ label: 'Timelines', value: parsed.symptoms.timelines.join('; ') }]
      : []),
    { label: 'Pertinent Negatives', value: parsed.symptoms.negatives || 'None noted', isMuted: !parsed.symptoms.negatives }
  ];
  leftY = drawSectionCard(leftX, leftY, colWidth, 'SYMPTOMS & TIMELINE', symptomFields, 'Patient Reported');

  const vitalFields = [
    { label: 'Blood Pressure', value: parsed.vitals.bloodPressure || 'Not provided', isMuted: !parsed.vitals.bloodPressure },
    { label: 'Heart Rate', value: parsed.vitals.heartRate || 'Not provided', isMuted: !parsed.vitals.heartRate },
    { label: 'Temperature', value: parsed.vitals.temperature || 'Not provided', isMuted: !parsed.vitals.temperature },
    { label: 'Oxygen Saturation', value: parsed.vitals.oxygenSaturation || 'Not provided', isMuted: !parsed.vitals.oxygenSaturation }
  ];
  leftY = drawSectionCard(leftX, leftY, colWidth, 'VITAL SIGNS', vitalFields, 'Self-Reported');

  // RIGHT COLUMN: Medical History + Medications & Allergies + Lifestyle
  const rightX = margin + colWidth + 14;
  const historyFields = [
    { label: 'Conditions', value: parsed.medicalHistory.conditions || 'None reported' },
    { label: 'Surgeries', value: parsed.medicalHistory.surgeries || 'None reported' },
    { label: 'Hospitalizations', value: parsed.medicalHistory.hospitalizations || 'None reported' },
    { label: 'Family History', value: parsed.medicalHistory.familyHistory || 'Not reported' }
  ];
  rightY = drawSectionCard(rightX, rightY, colWidth, 'MEDICAL & SURGICAL HISTORY', historyFields, 'Past History');

  const medFields = [
    { label: 'Current Meds', value: parsed.medications.current || 'None reported' },
    { label: 'Drug Allergies', value: parsed.allergies.drug || 'None reported' },
    { label: 'Environmental', value: parsed.allergies.environmental || 'None reported' }
  ];
  rightY = drawSectionCard(rightX, rightY, colWidth, 'MEDICATIONS & ALLERGIES', medFields, 'Active List');

  const lifestyleFields = [
    { label: 'Smoking', value: parsed.lifestyle.smoking || 'Not asked' },
    { label: 'Alcohol', value: parsed.lifestyle.alcohol || 'Not asked' },
    { label: 'Recent Travel', value: parsed.lifestyle.recentTravel || 'Not asked' },
    { label: 'Sick Contacts', value: parsed.lifestyle.sickContacts || 'Not asked' }
  ];
  rightY = drawSectionCard(rightX, rightY, colWidth, 'LIFESTYLE & HABITS', lifestyleFields, 'Social Hx');

  currentY = Math.min(leftY, rightY) - 4;

  // --- CLINICAL SUMMARY BOX (AI-Assisted Synthesis) ---
  const summaryText = parsed.clinicalSummary || 'Patient-reported pre-consultation information collected. Awaiting attending physician clinical examination and evaluation.';
  const summaryLines = wrapText(summaryText, usableWidth - 16, 7.8);
  const summaryBoxHeight = 16 + summaryLines.length * 10 + 6;

  checkPageBreak(summaryBoxHeight + 60);

  // Summary box
  drawRect(margin, currentY - summaryBoxHeight, usableWidth, summaryBoxHeight, [0.98, 0.98, 0.99], [0.82, 0.86, 0.92], 0.8);
  // Summary header
  drawRect(margin, currentY - 14, usableWidth, 14, [0.90, 0.94, 0.98], [0.80, 0.85, 0.92], 0.5);
  drawText('CLINICAL SUMMARY (AI-GENERATED FOR PHYSICIAN REVIEW)', margin + 8, currentY - 10, '/F1', 7.5, [0.08, 0.25, 0.48]);
  drawText('Pre-Consultation Synthesis', margin + usableWidth - 110, currentY - 10, '/F3', 6.8, [0.40, 0.48, 0.58]);

  summaryLines.forEach((sLine, sIdx) => {
    drawText(sLine, margin + 8, currentY - 24 - sIdx * 10, '/F2', 7.8, [0.15, 0.20, 0.28]);
  });

  currentY -= (summaryBoxHeight + 8);

  // --- FOOTER & CLINICAL DISCLAIMER (Fixed at page bottom) ---
  const footerY = 52;
  drawLine(margin, footerY + 18, margin + usableWidth, footerY + 18, [0.80, 0.84, 0.90], 0.75);

  const disclaimer1 = 'Clinical Notice: This document is an AI-assisted intake summary compiled from patient-reported symptoms and history.';
  const disclaimer2 = 'It does not establish a physician-patient relationship or constitute a definitive medical diagnosis. In a medical emergency, call 108 or 112.';
  drawText(disclaimer1, margin, footerY + 9, '/F3', 6.5, [0.40, 0.45, 0.55]);
  drawText(disclaimer2, margin, footerY, '/F3', 6.5, [0.40, 0.45, 0.55]);

  // Page numbering and generation stamp
  const pageStamp = `Page 1 of ${pages.length + 1}  |  Generated via Multi-Agent Medical Assistant  |  ${dateStr}`;
  drawText(pageStamp, margin + usableWidth - 230, footerY, '/F2', 6.5, [0.45, 0.50, 0.60]);

  finishPage();

  // --- CONSTRUCT PDF 1.4 BINARY STREAM ---
  const pdfParts = ['%PDF-1.4\n'];
  const offsets = [];

  function addPart(str) {
    // Calculate current byte offset using latin1 encoding
    offsets.push(pdfParts.reduce((acc, p) => acc + p.length, 0));
    pdfParts.push(str);
  }

  const numPages = pages.length;
  // Obj 1: Catalog
  addPart('1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n');

  // Obj 2: Pages
  const kidsStr = Array.from({ length: numPages }, (_, i) => `${3 + i * 2} 0 R`).join(' ');
  addPart(`2 0 obj\n<< /Type /Pages /Kids [${kidsStr}] /Count ${numPages} >>\nendobj\n`);

  // Font object IDs
  const fBoldId = 3 + numPages * 2;
  const fRegId = fBoldId + 1;
  const fItalicId = fBoldId + 2;

  // Pages & Streams
  pages.forEach((streamContent, idx) => {
    const pageObjId = 3 + idx * 2;
    const streamObjId = pageObjId + 1;
    const streamBytes = unescape(encodeURIComponent(streamContent)); // Byte-accurate length

    // Page Object
    addPart(
      `${pageObjId} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595.28 841.89] /Contents ${streamObjId} 0 R /Resources << /Font << /F1 ${fBoldId} 0 R /F2 ${fRegId} 0 R /F3 ${fItalicId} 0 R >> >> >>\nendobj\n`
    );

    // Stream Object
    addPart(
      `${streamObjId} 0 obj\n<< /Length ${streamContent.length} >>\nstream\n${streamContent}\nendstream\nendobj\n`
    );
  });

  // Fonts
  addPart(`${fBoldId} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n`);
  addPart(`${fRegId} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n`);
  addPart(`${fItalicId} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>\nendobj\n`);

  // Cross Reference Table
  const xrefOffset = pdfParts.reduce((acc, p) => acc + p.length, 0);
  let xref = `xref\n0 ${offsets.length + 1}\n0000000000 65535 f \n`;
  for (const o of offsets) {
    xref += `${String(o).padStart(10, '0')} 00000 n \n`;
  }

  const trailer = `trailer\n<< /Size ${offsets.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;
  const finalPdfString = pdfParts.join('') + xref + trailer;

  // Convert to binary Uint8Array
  const buffer = new Uint8Array(finalPdfString.length);
  for (let i = 0; i < finalPdfString.length; i++) {
    buffer[i] = finalPdfString.charCodeAt(i) & 0xff;
  }
  return buffer;
}

/**
 * Trigger immediate browser download of the clinical report as a PDF.
 * 
 * @param {string} reportText - The clinical report raw text
 * @param {Object} metadata - Optional session metadata
 * @returns {Object} Download result with status, filename, and byte size
 */
export function downloadClinicalReportPdf(reportText, metadata = {}) {
  if (!reportText || !reportText.trim()) {
    throw new Error('No clinical report data available to download.');
  }

  try {
    const pdfBuffer = buildClinicalReportPdf(reportText, metadata);
    const blob = new Blob([pdfBuffer], { type: 'application/pdf' });
    const dateStr = new Date().toISOString().slice(0, 10);
    const filename = `Clinical_Consultation_Report_${dateStr}.pdf`;

    const blobUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();

    setTimeout(() => {
      document.body.removeChild(link);
      URL.revokeObjectURL(blobUrl);
    }, 250);

    return {
      success: true,
      filename,
      size: pdfBuffer.length
    };
  } catch (err) {
    console.error('PDF Generation Error:', err);
    throw new Error(`Failed to generate clinical report PDF: ${err.message || 'Unknown error'}`);
  }
}
