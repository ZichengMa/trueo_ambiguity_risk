const puppeteer = require('puppeteer-core');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1500, height: 1200, deviceScaleFactor: 2 });

  const filePath = path.resolve(__dirname, 'poster.html').replace(/\\/g, '/');
  await page.goto(`file:///${filePath}`, { waitUntil: 'networkidle0' });

  // Wait for Google Fonts to render
  await new Promise(r => setTimeout(r, 2000));

  // Remove JS scaling so poster renders at its natural 1500×1200
  await page.evaluate(() => {
    const p = document.getElementById('poster');
    p.style.transform = 'none';
    p.style.transformOrigin = 'unset';
    document.body.style.padding = '0';
    document.body.style.background = 'white';
    document.body.style.alignItems = 'unset';
  });

  await page.pdf({
    path: path.resolve(__dirname, 'poster.pdf'),
    width: '1500px',
    height: '1200px',
    printBackground: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 }
  });

  await browser.close();
  console.log('Done → poster.pdf');
})();
