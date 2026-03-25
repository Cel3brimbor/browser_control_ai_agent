const BLOCKED_KEY = 'blockedUrls';
const WHITELIST_KEY = 'whitelistUrls';
const LMSTUDIO_CONFIG_KEY = 'lmstudioConfig';

async function getBlocked() {
  const { [BLOCKED_KEY]: list = [] } = await chrome.storage.sync.get(BLOCKED_KEY);
  return new Set(list);
}

async function getWhitelisted() {
  const { [WHITELIST_KEY]: list = [] } = await chrome.storage.sync.get(WHITELIST_KEY);
  return new Set(list);
}

async function addBlocked(url)   { await _addToSet(BLOCKED_KEY, url); }
async function addWhitelisted(url) { await _addToSet(WHITELIST_KEY, url); }
async function removeBlocked(url) { await _removeFromSet(BLOCKED_KEY, url); }
async function removeWhitelisted(url) { await _removeFromSet(WHITELIST_KEY, url); }

async function _addToSet(key, url) {
  const obj = await chrome.storage.sync.get(key);
  const set = new Set(obj[key] ?? []);
  set.add(url);
  await chrome.storage.sync.set({ [key]: [...set] });
}

async function _removeFromSet(key, url) {
  const obj = await chrome.storage.sync.get(key);
  const set = new Set(obj[key] ?? []);
  set.delete(url);
  await chrome.storage.sync.set({ [key]: [...set] });
}

async function getLmstudioConfig() {
  const { [LMSTUDIO_CONFIG_KEY]: config = {} } = await chrome.storage.sync.get(LMSTUDIO_CONFIG_KEY);
  return {
    baseUrl: config.baseUrl || 'http://127.0.0.1:1234',
    model: config.model || 'local-model'
  };
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'block-link',
    title: 'Block this link',
    contexts: ['link']
  });
  chrome.contextMenus.create({
    id: 'whitelist-link',
    title: 'Whitelist this link',
    contexts: ['link']
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!info.linkUrl) return;
  if (info.menuItemId === 'block-link') {
    await addBlocked(info.linkUrl);
  } else if (info.menuItemId === 'whitelist-link') {
    await addWhitelisted(info.linkUrl);
  }
  chrome.runtime.sendMessage({ type: 'REFRESH_UI' });
});

chrome.runtime.onMessage.addListener(async (msg, sender, sendResponse) => {
  if (msg.type === 'ADD_BLOCK') await addBlocked(msg.url);
  if (msg.type === 'REMOVE_BLOCK') await removeBlocked(msg.url);
  if (msg.type === 'ADD_WHITELIST') await addWhitelisted(msg.url);
  if (msg.type === 'REMOVE_WHITELIST') await removeWhitelisted(msg.url);
  if (msg.type === 'SET_LMSTUDIO_CONFIG') await chrome.storage.sync.set({ [LMSTUDIO_CONFIG_KEY]: msg.config });
  if (msg.type === 'REFRESH_UI') chrome.runtime.sendMessage({ type: 'REFRESH_UI' });
});

async function handleNavigation(details) {
  if (details.frameId !== 0) return;

  const url = details.url;
  const tabId = details.tabId;

  console.log('navigation event', { type: details.transitionType || details.reason || 'unknown', url, tabId });

  const blocked = await getBlocked();
  const whitelisted = await getWhitelisted();

  //blacklist explicit block
  if (blocked.has(url) || blocked.has(new URL(url).origin + '/*')) {
    explicitBlock(tabId);
    return;
  }

  //no agent check if whitelist
  if (whitelisted.has(url) || whitelisted.has(new URL(url).origin + '/*')) {
    console.log('whitelisted url, allowing:', url);
    return;
  }

  //check and validate only http/https
  if (!/^https?:/.test(url)) return;

  try {
    const tab = await chrome.tabs.get(tabId);
    const allTabs = await chrome.tabs.query({});
    const otherTabs = allTabs.filter(t => t.id !== tabId && t.title);
    const currentTitles = otherTabs.map(t => t.title);

    if (currentTitles.length < 4) {
      //not enough context so no block 
      return;
    }

    const config = await getLmstudioConfig();
    const allowed = await globalThis.agentDetermination(url, tab.title || '', currentTitles, config);
    if (!allowed) {
      ai_block(tabId);
    }
  } catch (err) {
    console.error('AI decision failed:', err);
  }
}

chrome.webNavigation.onCommitted.addListener(handleNavigation);
chrome.webNavigation.onHistoryStateUpdated.addListener(handleNavigation);

function ai_block(tabId) {
  chrome.tabs.update(tabId, { url: chrome.runtime.getURL('blocked_by_ai.html') });
}

function explicitBlock(tabId) {
  chrome.tabs.update(tabId, { url: chrome.runtime.getURL('explicit_block.html') });
}

async function agentDetermination(newUrl, newTitle, currentTabTitles, config) {
  // if (/^https?:\/\/(www\.)?youtube\.com/i.test(newUrl)) {
  //   console.log('Bypassing AI block for YouTube URL', newUrl);
  //   return true;
  // }

  const prompt = `
These are the current tab titles in the user's browser: ${currentTabTitles.join(', ')}.
The current request is opening URL: ${newUrl} with title: "${newTitle}".
Does this new tab seem on-task relative to the existing tabs?
Answer only exactly one of: on-task or off-task.
Also include a brief reason in plain text.`;

  const payload = {
    model: config.model,
    messages: [{ role: 'user', content: prompt }],
    temperature: 0.0,
    max_tokens: 100
  };

  const apiUrl = `${config.baseUrl}/v1/chat/completions`;

  try {
    const resp = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const data = await resp.json();
    const answerRaw = data.choices?.[0]?.message?.content?.trim() || '';

    const normalized = answerRaw.toLowerCase();
    const isOnTask = /\b(on[- ]task|on task|on-task|1)\b/.test(normalized);
    const isOffTask = /\b(off[- ]task|off task|off-task|0)\b/.test(normalized);

    if (isOnTask) {
      console.log('Decision: allowed (on-task)');
      return true;
    }
    if (isOffTask) {
      console.log('Decision: blocked (off-task)');
      return false;
    }

    console.warn('Unexpected LM Studio answer, defaulting to safe allow:', answerRaw);
    return true;
  } catch (err) {
    console.error('LM Studio error:', err);
    return false;
  }
}