const puppeteer = require('puppeteer');
const fs = require('fs-extra');
const path = require('path');

const BASE_URL = 'https://echarts.apache.org/examples/zh/index.html';
const RAW_DIR = path.join(__dirname, '../data/raw/echarts');
const SCREENSHOT_DIR = path.join(RAW_DIR, 'screenshots');
const CODE_DIR = path.join(RAW_DIR, 'codes');

fs.ensureDirSync(SCREENSHOT_DIR);
fs.ensureDirSync(CODE_DIR);

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });

  console.log('正在访问首页...');
  await page.goto(BASE_URL, { waitUntil: 'networkidle2' });
  await page.waitForSelector('.example-list-panel .example-list-item a.example-link', { timeout: 15000 });

  const examples = await page.evaluate(() => {
    const links = Array.from(document.querySelectorAll('.example-list-panel .example-list-item a.example-link'));
    return links.map(a => ({
      title: a.querySelector('img')?.alt || 'untitled',
      href: a.href,
      codeParam: a.href.split('?c=')[1]
    }));
  });

  console.log(`共找到 ${examples.length} 个示例`);

  for (let i = 0; i < examples.length; i++) {
    const example = examples[i];
    console.log(`[${i + 1}/${examples.length}] 处理: ${example.title} (${example.codeParam})`);

    const detailPage = await browser.newPage();
    await detailPage.setViewport({ width: 1280, height: 800 });

    try {
      const detailUrl = `https://echarts.apache.org/examples/zh/editor.html?c=${example.codeParam}`;
      await detailPage.goto(detailUrl, { waitUntil: 'networkidle2' });

      // ----- 截图部分：处理 iframe 内的图表 -----
      console.log('  等待 iframe 加载...');
      // 等待 iframe 出现（右侧图表区域的 iframe）
      const iframeElement = await detailPage.waitForSelector('#chart-panel iframe', { timeout: 10000 });
      // 获取 iframe 的 content frame
      const chartFrame = await iframeElement.contentFrame();
      if (!chartFrame) throw new Error('无法获取 iframe 内容');

      // 在 iframe 内等待 canvas 出现（图表渲染完成的标志）
      console.log('  等待图表渲染...');
      await chartFrame.waitForSelector('canvas', { timeout: 10000 });

      // 额外等待一小段时间确保图表完全稳定
      await new Promise(resolve => setTimeout(resolve, 500));

      // 对 iframe 内的图表容器截图（使用 #chart-container 或 canvas 的父级）
      // 优先选择 #chart-container，如果没有则直接对 canvas 截图
      let chartContainer = await chartFrame.$('#chart-container');
      if (!chartContainer) {
        chartContainer = await chartFrame.$('canvas');
      }
      if (chartContainer) {
        const screenshotPath = path.join(SCREENSHOT_DIR, `example_${i + 1}.png`);
        await chartContainer.screenshot({ path: screenshotPath });
        console.log(`  截图已保存: example_${i + 1}.png`);
      } else {
        console.log('  警告: 未找到图表元素，跳过截图');
      }

      // ----- 代码提取部分：从左侧 ACE 编辑器获取代码 -----
      console.log('  尝试从 ACE 编辑器获取代码...');
      const code = await detailPage.evaluate(() => {
        // 方法1: 通过 ACE 编辑器实例获取
        // 尝试多种可能的全局变量
        if (window.ace) {
          const editor = window.ace.edit('code-panel');
          if (editor) return editor.getValue();
        }
        if (window.editor) {
          return window.editor.getValue();
        }
        // 方法2: 如果编辑器实例未暴露，从 DOM 行拼接（兜底方案）
        const lines = Array.from(document.querySelectorAll('.ace_line'));
        if (lines.length > 0) {
          return lines.map(line => line.textContent).join('\n');
        }
        // 方法3: 尝试获取隐藏的 textarea（某些情况下可能有效）
        const textarea = document.querySelector('.ace_text-input');
        if (textarea && textarea.value) {
          return textarea.value;
        }
        return null;
      });

      if (code) {
        const codePath = path.join(CODE_DIR, `example_${i + 1}.js`);
        fs.writeFileSync(codePath, code);
        console.log(`  代码已保存: example_${i + 1}.js (长度 ${code.length})`);
      } else {
        console.log('  警告: 未能获取到代码');
        // 可选：保存一个空文件标记失败
        fs.writeFileSync(path.join(CODE_DIR, `example_${i + 1}.js`), '// 代码获取失败');
      }

    } catch (err) {
      console.error(`  处理示例时出错: ${err.message}`);
    } finally {
      await detailPage.close();
    }

    // 随机延迟避免请求过频
    await new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 1000));
  }

  await browser.close();
  console.log('全部任务完成！');
})();