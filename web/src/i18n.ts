// UI language from the browser: German when it prefers German, English otherwise. `?lang=de|en` overrides.
const nav = typeof navigator === 'undefined' ? undefined : navigator;
const forced = typeof location === 'undefined' ? null : new URLSearchParams(location.search).get('lang');
export const LANG: 'de' | 'en' = forced === 'de' || forced === 'en' ? forced
  : (nav?.languages?.[0] ?? nav?.language ?? 'en').toLowerCase().startsWith('de') ? 'de' : 'en';

export const tr = (de: string, en: string): string => (LANG === 'de' ? de : en);

/** Static markup is English; elements carry their German in data-de (text) and data-de-title (tooltip). */
export function localizeDom(root: ParentNode = document): void {
  document.documentElement.lang = LANG;
  if (LANG !== 'de') return;
  root.querySelectorAll<HTMLElement>('[data-de]').forEach((el) => { el.textContent = el.dataset.de!; });
  root.querySelectorAll<HTMLElement>('[data-de-title]').forEach((el) => { el.title = el.dataset.deTitle!; });
  const t = document.querySelector('title');
  if (t?.dataset.de) document.title = t.dataset.de;
}
