/**
 * Stream Client for Server-Sent Events (SSE) over HTTP POST
 * Consumes /chat/stream and provides progressive callbacks.
 */

export async function streamChat({
  query,
  workspace = 'conversation',
  sessionId = null,
  onStart = () => {},
  onStatus = () => {},
  onToken = () => {},
  onComplete = () => {},
  onError = () => {},
  signal = null
}) {
  const headers = {
    'Content-Type': 'application/json'
  };
  if (sessionId) {
    headers['X-Session-ID'] = sessionId;
  }

  let response;
  try {
    response = await fetch('/chat/stream', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        query,
        workspace,
        workspace_type: workspace,
        conversation_history: []
      }),
      credentials: 'include',
      signal
    });
  } catch (fetchErr) {
    if (signal && signal.aborted) {
      return { aborted: true };
    }
    const err = new Error(fetchErr.message || 'Network connection failed');
    onError(err);
    throw err;
  }

  if (!response.ok) {
    let errorDetail = `Server returned status ${response.status}`;
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || errJson.message || errorDetail;
    } catch (_) {}
    const err = new Error(errorDetail);
    onError(err);
    throw err;
  }

  // Extract session ID from response headers if returned
  const returnedSessionId = response.headers.get('X-Session-ID') || sessionId;

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let currentEvent = 'message';
  let completePayload = null;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      // Keep the last partial line in the buffer
      buffer = lines.pop() || '';

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) {
          // Empty line indicates event boundary in SSE
          currentEvent = 'message';
          continue;
        }

        if (line.startsWith('event:')) {
          currentEvent = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          const rawData = line.slice(5).trim();
          try {
            const parsed = JSON.parse(rawData);
            if (currentEvent === 'start') {
              onStart(parsed);
            } else if (currentEvent === 'status') {
              onStatus(parsed);
            } else if (currentEvent === 'token') {
              const chunk = parsed.chunk || '';
              if (chunk) {
                onToken(chunk);
              }
            } else if (currentEvent === 'complete') {
              completePayload = parsed;
              onComplete(parsed);
            } else if (currentEvent === 'error') {
              const errMsg = parsed.message || 'Stream processing error';
              onError(new Error(errMsg));
            }
          } catch (jsonErr) {
            console.warn('[StreamClient] Failed to parse SSE JSON data:', rawData, jsonErr);
          }
        }
      }
    }

    return {
      success: true,
      sessionId: returnedSessionId,
      completePayload
    };
  } catch (readErr) {
    if (signal && signal.aborted) {
      return { aborted: true };
    }
    console.error('[StreamClient] Stream read error:', readErr);
    onError(readErr);
    throw readErr;
  } finally {
    try {
      reader.releaseLock();
    } catch (_) {}
  }
}
