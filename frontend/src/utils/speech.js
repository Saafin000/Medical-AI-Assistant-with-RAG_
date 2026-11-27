/**
 * Utility functions for Speech-to-Text and Text-to-Speech
 */

/**
 * Decode common HTML and numeric character entities so TTS never reads raw entity codes.
 */
function decodeSpeechEntities(text) {
  if (!text) return '';
  let decoded = String(text);
  for (let i = 0; i < 2; i += 1) {
    const next = decoded
      .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => {
        const code = parseInt(hex, 16);
        return Number.isNaN(code) ? '' : String.fromCodePoint(code);
      })
      .replace(/&#(\d+);/g, (_, dec) => {
        const code = parseInt(dec, 10);
        return Number.isNaN(code) ? '' : String.fromCodePoint(code);
      })
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&#39;|&apos;/g, "'")
      .replace(/&nbsp;/g, ' ');
    if (next === decoded) break;
    decoded = next;
  }
  return decoded;
}

/**
 * Clean markdown and truncate text for natural Text-to-Speech delivery
 * Strips code blocks, links, formatting markers, headers, HTML tags/entities, and excessive whitespace.
 * Truncates to 1000 characters as in the original application.
 * 
 * @param {string} text - Raw markdown text
 * @returns {string} Clean plain text suitable for TTS
 */
export function cleanTextForSpeech(text) {
  if (!text) return '';

  let cleaned = decodeSpeechEntities(text);

  // Strip any residual HTML tags without executing them
  cleaned = cleaned.replace(/<[^>]*>/g, ' ');

  // Remove code blocks
  cleaned = cleaned.replace(/```[\s\S]*?```/g, 'Code block omitted for speech.');

  // Remove inline code
  cleaned = cleaned.replace(/`([^`]+)`/g, '$1');

  // Remove horizontal dividers (---, ===, ***)
  cleaned = cleaned.replace(/^\s*([-=*_])\1{2,}\s*$/gm, ' ');

  // Remove headers
  cleaned = cleaned.replace(/#{1,6}\s+/g, '');

  // Remove bold/italic markers
  cleaned = cleaned.replace(/(\*\*|__)(.*?)\1/g, '$2');
  cleaned = cleaned.replace(/(\*|_)(.*?)\1/g, '$2');

  // Remove list markers
  cleaned = cleaned.replace(/^\s*[\*\-+]\s+/gm, '');
  cleaned = cleaned.replace(/^\s*\d+\.\s+/gm, '');

  // Remove links, keep text
  cleaned = cleaned.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');

  // Remove images
  cleaned = cleaned.replace(/!\[([^\]]+)\]\([^)]+\)/g, 'Image: $1');

  // Remove blockquotes
  cleaned = cleaned.replace(/^\>\s+/gm, '');

  // Handle line breaks and excessive whitespace sensibly for speech
  cleaned = cleaned.replace(/\n{2,}/g, '. ');
  cleaned = cleaned.replace(/\n/g, ' ');
  cleaned = cleaned.replace(/\s{2,}/g, ' ');

  // Truncate to 1000 characters to keep requests within reasonable limits
  if (cleaned.length > 1000) {
    cleaned = cleaned.substring(0, 1000) + '...';
  }

  return cleaned.trim();
}

/**
 * Default ElevenLabs Voice ID (Sarah - premade voice supported across all tiers)
 */
export const DEFAULT_VOICE_ID = 'EXAVITQu4vr4xnSDxMaL';
