const BLOCKED_KEY = 'blockedUrls';
const WHITELIST_KEY = 'whitelistUrls';
const GEMINI_CONFIG_KEY = 'geminiConfig';

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

async function getGeminiConfig() {
  const { [GEMINI_CONFIG_KEY]: config = {} } = await chrome.storage.sync.get(GEMINI_CONFIG_KEY);
  return {
    projectId: config.projectId || 'ai-browser-blocker', //from google cloud
    location: config.location || 'us-central1',
    model: config.model || 'google/gemini-2.0-flash-001'
  };
}

async function getAccessToken() {
  try {
    const token = await new Promise((resolve, reject) => {
      chrome.identity.getAuthToken({ interactive: true }, (token) => {
        if (chrome.runtime.lastError) reject(chrome.runtime.lastError);
        else resolve(token);
      });
    });
    return token;
  } catch (err) {
    console.error('OAuth error:', err);
    throw err;
  }
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
  if (msg.type === 'SET_GEMINI_CONFIG') await chrome.storage.sync.set({ [GEMINI_CONFIG_KEY]: msg.config });
  if (msg.type === 'REFRESH_UI') chrome.runtime.sendMessage({ type: 'REFRESH_UI' });
});

chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  if (details.frameId !== 0) return;

  const url = details.url;
  const tabId = details.tabId;

  const blocked = await getBlocked();
  const whitelisted = await getWhitelisted();

  //blacklist explicit block 
  if (blocked.has(url) || blocked.has(new URL(url).origin + '/*')) {
    explicitBlock(tabId);
    return;
  }

  //no agent check if whitelist
  if (whitelisted.has(url) || whitelisted.has(new URL(url).origin + '/*')) {
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

    const config = await getGeminiConfig();
    const token = await getAccessToken();
    const allowed = await globalThis.agentDetermination(url, tab.title || '', currentTitles, config, token);
    if (!allowed) {
      ai_block(tabId);
    }
  } catch (err) {
    console.error('AI decision failed:', err);
    //redirectToBlocked(tabId); 
  }
});

function ai_block(tabId) {
  chrome.tabs.update(tabId, { url: chrome.runtime.getURL('blocked_by_ai.html') });
}

function explicitBlock(tabId) {
  chrome.tabs.update(tabId, { url: chrome.runtime.getURL('explicit_block.html') });
}

async function agentDetermination(newUrl, newTitle, currentTabTitles, config, accessToken) {
  const prompt = `
  These are the current tab titles in the user's browser: ${currentTabTitles.join(', ')}. They are comma-seperated.
  Does the newly opened tab's title "${newTitle}" seem to align or be on-task compared to the current ones?
  Return 0 or 1 only. 0 for off-task, 1 for on-task.`;

  const payload = {
    model: config.model,
    messages: [{ role: 'user', content: prompt }],
    temperature: 0.0,
    max_tokens: 10
  };

  const apiUrl = `https://${config.location}-aiplatform.googleapis.com/v1/projects/${config.projectId}/locations/${config.location}/endpoints/openapi/chat/completions`;

  try {
    const resp = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`
      },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const data = await resp.json();
    const answer = data.choices?.[0]?.message?.content?.trim();

    return answer === '1';
  } catch (err) {
    console.error('Gemini error:', err);
    return false;
  }
}