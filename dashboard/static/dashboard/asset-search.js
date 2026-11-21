const assetSearch = document.querySelector('[data-asset-search]');

if (assetSearch) {
  const input = assetSearch.querySelector('input[name="symbol"]');
  const list = document.getElementById('asset-search-results');
  const status = document.getElementById('asset-search-status');
  const remembered = new Map();
  let timer;
  let controller;
  let version = 0;
  let results = [];
  let active = -1;

  function close() {
    list.hidden = true;
    input.setAttribute('aria-expanded', 'false');
    input.removeAttribute('aria-activedescendant');
    active = -1;
  }

  function cancel() {
    clearTimeout(timer);
    controller?.abort();
    version += 1;
    input.removeAttribute('aria-busy');
  }

  function select(index) {
    const asset = results[index];
    if (!asset) return;
    cancel();
    input.value = asset.symbol;
    close();
    status.textContent = `${asset.name} (${asset.symbol}) selected.`;
    input.focus();
  }

  function show(items, message = '') {
    results = items;
    list.replaceChildren();
    active = -1;
    input.removeAttribute('aria-activedescendant');
    items.forEach((asset, index) => {
      const option = document.createElement('div');
      option.id = `asset-option-${index}`;
      option.className = 'asset-option';
      option.setAttribute('role', 'option');
      option.setAttribute('aria-selected', 'false');
      const name = document.createElement('span');
      name.className = 'asset-option-name';
      name.textContent = asset.name;
      const details = document.createElement('span');
      details.className = 'asset-option-details';
      details.textContent = [asset.symbol, asset.type, asset.exchange, asset.currency].filter(Boolean).join(' · ');
      option.append(name, details);
      option.addEventListener('pointerdown', event => event.preventDefault());
      option.addEventListener('click', () => select(index));
      list.append(option);
    });
    list.hidden = items.length === 0;
    input.setAttribute('aria-expanded', String(items.length > 0));
    status.textContent = message || (items.length ? 'Select an asset below.' : 'No matches found. Try another name or symbol.');
  }

  async function search(query, currentVersion) {
    controller = new AbortController();
    const request = controller;
    const timeout = setTimeout(() => request.abort(), 12000);
    try {
      const url = new URL(assetSearch.dataset.searchUrl, window.location.origin);
      url.searchParams.set('q', query);
      const response = await fetch(url, {signal: request.signal, headers: {'Accept': 'application/json'}});
      if (!response.ok || response.redirected) throw new Error('Search unavailable');
      const data = await response.json();
      if (currentVersion !== version || document.activeElement !== input) return;
      if (!Array.isArray(data.results)) throw new Error('Invalid search results');
      if (!data.message) remembered.set(query, data);
      show(data.results, data.message);
    } catch (error) {
      if (currentVersion !== version || document.activeElement !== input) return;
      close();
      status.textContent = 'Search is unavailable right now. You can still enter an asset symbol.';
    } finally {
      clearTimeout(timeout);
      if (currentVersion === version) input.removeAttribute('aria-busy');
    }
  }

  function schedule() {
    cancel();
    close();
    results = [];
    const query = input.value.trim().toLowerCase();
    if (query.length < 2) {
      status.textContent = query ? 'Type at least two letters to search.' : '';
      return;
    }
    if (remembered.has(query)) {
      const data = remembered.get(query);
      show(data.results, data.message);
      return;
    }
    status.textContent = 'Searching…';
    input.setAttribute('aria-busy', 'true');
    const currentVersion = version;
    timer = setTimeout(() => search(query, currentVersion), 350);
  }

  input.addEventListener('input', event => {
    if (!event.isComposing) schedule();
  });
  input.addEventListener('compositionend', schedule);
  input.addEventListener('focus', () => {
    if (input.value.trim()) schedule();
  });
  input.addEventListener('blur', () => {
    cancel();
    close();
    if (status.textContent === 'Searching…') status.textContent = '';
  });
  input.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      cancel();
      close();
      status.textContent = '';
      return;
    }
    if (list.hidden || event.isComposing) return;
    if (event.key === 'Enter') {
      event.preventDefault();
      select(active < 0 ? 0 : active);
      return;
    }
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    event.preventDefault();
    active = event.key === 'ArrowDown' ? (active + 1) % results.length : (active <= 0 ? results.length - 1 : active - 1);
    Array.from(list.children).forEach((option, index) => option.setAttribute('aria-selected', String(index === active)));
    input.setAttribute('aria-activedescendant', list.children[active].id);
    list.children[active].scrollIntoView({block: 'nearest'});
  });
}
