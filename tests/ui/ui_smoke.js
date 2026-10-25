const { chromium } = require('playwright');
const S = process.argv[2];
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1500 } });
  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  await page.goto('http://127.0.0.1:7861/', { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('text=Inspect & open project', { timeout: 60000 });
  // new video through the real path: inspect -> run -> expect actionable Whisper error
  await page.getByLabel('Video file (path on this computer)').fill(S + '/demo2.mp4');
  await page.getByLabel('Output folder (projects are stored here)').fill(S + '/uiproj');
  await page.getByRole('button', { name: '🔍 Inspect & open project' }).click();
  await page.waitForSelector('text=Project folder', { timeout: 30000 });
  await page.getByRole('button', { name: '▶ Run / resume analysis' }).click();
  await page.waitForSelector('text=not downloaded', { timeout: 60000 });
  await page.screenshot({ path: S + '/ui5_error.png', fullPage: true });
  console.log('error shown:', (await page.locator('.clp-err').first().innerText()).replace(/\n/g, ' | '));
  // diagnostics tab
  await page.getByRole('tab', { name: 'Diagnostics' }).click();
  await page.getByRole('button', { name: /Check this computer/ }).click();
  await page.waitForFunction(() => [...document.querySelectorAll('textarea')].some(t => t.value.includes('Recommended')), null, { timeout: 30000 });
  console.log('diag ok');
  // reopen seeded project, approve, export
  await page.getByRole('tab', { name: '1 · Import & process' }).click();
  await page.getByLabel('…or reopen a project').click();
  await page.getByRole('option', { name: /demo-/ }).first().click();
  await page.getByRole('button', { name: 'Open', exact: true }).click();
  await page.waitForTimeout(1500);
  await page.getByRole('tab', { name: '2 · Review clips' }).click();
  await page.waitForTimeout(1500);
  await page.getByText('The lights went out').last().click();
  await page.waitForSelector('video', { timeout: 90000 });
  await page.waitForTimeout(2500);
  await page.getByRole('button', { name: '✅ Approve' }).click();
  await page.waitForSelector('text=approved', { timeout: 10000 });
  console.log('approved shown');
  await page.screenshot({ path: S + '/ui6_approved.png', fullPage: true });
  await page.getByRole('tab', { name: '3 · Export' }).click();
  await page.getByRole('button', { name: '📤 Export approved clips' }).click();
  await page.waitForSelector('text=Exported 1 of 1', { timeout: 120000 });
  await page.screenshot({ path: S + '/ui7_export.png', fullPage: true });
  console.log('export:', (await page.locator('text=Exported 1 of 1').first().innerText()));
  console.log('PAGE ERRORS:', JSON.stringify(errors));
  await browser.close();
})().catch(e => { console.error('FAILED', e.message); process.exit(1); });
