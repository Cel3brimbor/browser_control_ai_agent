export async function shouldAllowWithGemini(newUrl, newTitle, currentTabTitles, config, accessToken) {
  // const prompt = `
  // These are the current tab titles in the user's browser: ${currentTabTitles.join(', ')}. They are comma-seperated.
  // Does the newly opened tab's title "${newTitle}" seem to align or be on-task compared to the current ones?
  // Return 0 or 1 only. 0 for off-task, 1 for on-task.`;


  const prompt = `
  These are the current tab titles in the user's browser: ${currentTabTitles.join(', ')}. They are comma-seperated.
  Does the newly opened tab's title "${newTitle}" seem to align or be on-task compared to the current ones?
  Return whether it is on task or off task. Give reason why.`;


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

    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);

    const data = await resp.json();
    const answer = data.choices?.[0]?.message?.content?.trim();

    return answer === '1';
  } catch (err) {
    console.error('Gemini error:', err);
    return false;
  }
}