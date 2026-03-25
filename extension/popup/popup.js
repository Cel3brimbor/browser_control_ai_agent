const baseUrlInput = document.getElementById('baseUrl');
const modelInput = document.getElementById('model');
const saveLmstudioConfigBtn = document.getElementById('saveLmstudioConfig');

const blockInput = document.getElementById('blockInput');
const addBlockBtn = document.getElementById('addBlock');
const blockList = document.getElementById('blockList');
const whiteInput = document.getElementById('whiteInput');
const addWhiteBtn = document.getElementById('addWhite');
const whiteList = document.getElementById('whiteList');

async function loadData() {
  const { blockedUrls = [], whitelistUrls = [], lmstudioConfig = {} } = await chrome.storage.sync.get([
    'blockedUrls', 'whitelistUrls', 'lmstudioConfig'
  ]);

  baseUrlInput.value = lmstudioConfig.baseUrl || 'http://127.0.0.1:1234';
  modelInput.value = lmstudioConfig.model || 'local-model';

  blockList.innerHTML = '';
  blockedUrls.forEach(url => addListItem(blockList, url, 'REMOVE_BLOCK'));

  whiteList.innerHTML = '';
  whitelistUrls.forEach(url => addListItem(whiteList, url, 'REMOVE_WHITELIST'));
}

async function checkAuthStatus() {
  try {
    const token = await new Promise((resolve) => chrome.identity.getAuthToken({ interactive: false }, resolve));
    authStatusEl.textContent = token ? 'Auth: OK' : 'Auth: Click Save to authenticate';
    authStatusEl.style.color = token ? 'green' : 'orange';
  } catch (err) {
    authStatusEl.textContent = 'Auth: Failed - Check console';
    authStatusEl.style.color = 'red';
  }
}

function addListItem(ul, url, removeType) {
  const li = document.createElement('li');
  li.innerHTML = `
    <span title="${url}">${truncate(url, 40)}</span>
    <button>x</button>
  `;
  li.querySelector('button').onclick = () => {
    chrome.runtime.sendMessage({ type: removeType, url });
  };
  ul.appendChild(li);
}

function truncate(str, n) {
  return str.length > n ? str.slice(0, n - 3) + '...' : str;
}

saveGeminiConfigBtn.onclick = async () => {
  const config = {
    projectId: projectIdInput.value.trim(),
    location: locationInput.value.trim(),
    model: modelInput.value.trim()
  };
  if (config.projectId) {
    await chrome.runtime.sendMessage({ type: 'SET_GEMINI_CONFIG', config });
    await checkAuthStatus(); 
  }
};

addBlockBtn.onclick = () => {
  const url = blockInput.value.trim();
  if (url) {
    chrome.runtime.sendMessage({ type: 'ADD_BLOCK', url });
    blockInput.value = '';
  }
};

addWhiteBtn.onclick = () => {
  const url = whiteInput.value.trim();
  if (url) {
    chrome.runtime.sendMessage({ type: 'ADD_WHITELIST', url });
    whiteInput.value = '';
  }
};

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'REFRESH_UI') loadData();
});

loadData();