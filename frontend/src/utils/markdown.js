import { marked } from 'marked';
import DOMPurify from 'dompurify';

// Configure marked options
marked.setOptions({
  gfm: true,
  breaks: true,
});

/**
 * Custom renderer to ensure external links open safely in a new tab
 */
const renderer = new marked.Renderer();
const originalLinkRenderer = renderer.link.bind(renderer);

renderer.link = function (href, title, text) {
  // Check if args are in object format (newer marked versions) or positional
  let url = href;
  let linkTitle = title;
  let linkText = text;

  if (typeof href === 'object' && href !== null) {
    url = href.href;
    linkTitle = href.title;
    linkText = href.text;
  }

  const titleAttr = linkTitle ? ` title="${linkTitle}"` : '';
  const safeUrl = (url || '').startsWith('javascript:') ? '#' : (url || '#');
  return `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer"${titleAttr} class="markdown-link">${linkText} <i class="fas fa-external-link-alt source-icon"></i></a>`;
};

marked.use({ renderer });

/**
 * Safely parse markdown string and sanitize the resulting HTML
 * @param {string} markdownText 
 * @returns {string} Sanitized HTML string
 */
export function renderMarkdown(markdownText) {
  if (!markdownText) return '';
  try {
    const rawHtml = marked.parse(markdownText);
    return DOMPurify.sanitize(rawHtml, {
      ADD_ATTR: ['target', 'rel'],
      ADD_TAGS: ['iframe'], // in case needed, otherwise standard tags
      FORBID_TAGS: ['script', 'style', 'input', 'form']
    });
  } catch (error) {
    console.error('Error rendering markdown:', error);
    // Fallback: simple text escaping
    const div = document.createElement('div');
    div.textContent = markdownText;
    return div.innerHTML;
  }
}
