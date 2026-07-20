// list-filter.js — in-place type filtering for the home page post list.
//
// The filter bar links point at the /categories/<type>/ term pages, which
// is what no-JS visitors and crawlers get. On the home page (the only page
// where the nav carries data-list-filter, and where the list is rendered
// unpaginated) we intercept clicks and hide/show rows instead.
//
// State lives in the URL hash (#comparisons) so a filtered view survives
// reload and can be shared; replaceState keeps the back button simple.

const bar = document.querySelector('[data-list-filter]');

if (bar) {
  const rows = Array.from(document.querySelectorAll('.tw-postrow[data-cat]'));
  const links = Array.from(bar.querySelectorAll('a[data-cat]'));
  const cmd = document.querySelector('[data-list-cmd]');
  const baseCmd = cmd ? cmd.textContent : '';

  const apply = (cat) => {
    rows.forEach((row) => {
      row.classList.toggle('tw-hide', cat !== 'all' && row.dataset.cat !== cat);
    });
    links.forEach((link) => {
      const active = link.dataset.cat === cat;
      link.classList.toggle('tw-filter-active', active);
      if (active) {
        link.setAttribute('aria-current', 'true');
      } else {
        link.removeAttribute('aria-current');
      }
    });
    // Terminal flavor: reflect the filter in the fake prompt command.
    if (cmd) cmd.textContent = cat === 'all' ? baseCmd : baseCmd + ' | grep ' + cat;
  };

  bar.addEventListener('click', (event) => {
    const link = event.target.closest('a[data-cat]');
    if (!link) return;
    event.preventDefault();
    const cat = link.dataset.cat;
    apply(cat);
    history.replaceState(null, '', cat === 'all' ? location.pathname + location.search : '#' + cat);
  });

  const initial = location.hash.slice(1);
  if (links.some((link) => link.dataset.cat === initial)) apply(initial);
}
