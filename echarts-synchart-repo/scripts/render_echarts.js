// render_echarts.js
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const CHROME_PATH = '/data/home/liyunzhe/chrome/chrome-linux64/chrome';

function hasAsyncRequest(content) {
    const asyncPatterns = [
        /\$\.get\s*\(/,
        /\$\.ajax\s*\(/,
        /fetch\s*\(/,
        /XMLHttpRequest/,
        /require\s*\(/
    ];
    return asyncPatterns.some(pattern => pattern.test(content));
}

function buildHTML(jsContent) {
    // 转义 JS 内容中的反引号和特殊字符，避免破坏模板字符串
    const escapedContent = jsContent.replace(/`/g, '\\`').replace(/\$/g, '\\$');
    
    return `
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="echarts.min.js"></script>
    <style>
        body { margin: 0; padding: 0; background: white; }
        #main { width: 800px; height: 600px; }
    </style>
</head>
<body>
    <div id="main"></div>
    <script>
        var ROOT_PATH = 'https://echarts.apache.org/examples';
        var myChart = null;
        var renderCompleted = false;

        window.addEventListener('load', function() {
            var chartDom = document.getElementById('main');
            if (!chartDom) {
                console.error('Chart container not found');
                window.renderCompleted = false;
                return;
            }
            myChart = echarts.init(chartDom);

            // 执行用户代码
            try {
                eval(${JSON.stringify(escapedContent)});
            } catch(e) {
                console.error('执行用户代码错误:', e);
            }

            // 如果用户定义了 option，则自动渲染
            if (typeof option !== 'undefined' && option !== null) {
                try {
                    myChart.setOption(option);
                    console.log('option 已设置');
                } catch(e) {
                    console.error('设置 option 错误:', e);
                }
            } else {
                console.warn('未找到 option，图表可能无法渲染');
            }

            // 检查 canvas 是否真的画出了东西
            var checkCount = 0;
            var maxChecks = 20;
            var checkInterval = 500;
            var intervalId = setInterval(function() {
                var canvas = document.querySelector('canvas');
                if (!canvas) {
                    console.warn('未找到 canvas 元素');
                    if (++checkCount >= maxChecks) {
                        console.error('超时：未出现 canvas');
                        clearInterval(intervalId);
                        window.renderCompleted = false;
                    }
                    return;
                }
                var ctx = canvas.getContext('2d');
                var w = canvas.width;
                var h = canvas.height;
                if (w === 0 || h === 0) {
                    if (++checkCount >= maxChecks) {
                        console.error('超时：canvas 尺寸为 0');
                        clearInterval(intervalId);
                        window.renderCompleted = false;
                    }
                    return;
                }
                try {
                    var imageData = ctx.getImageData(0, 0, w, h);
                    var nonWhite = 0;
                    for (var i = 0; i < imageData.data.length; i += 4) {
                        if (imageData.data[i] < 250 || imageData.data[i+1] < 250 || imageData.data[i+2] < 250) {
                            nonWhite++;
                        }
                    }
                    var total = w * h;
                    var ratio = nonWhite / total;
                    console.log('非白色像素比例: ' + (ratio * 100).toFixed(2) + '%');
                    if (ratio > 0.01) {
                        console.log('图表渲染有内容');
                        clearInterval(intervalId);
                        window.renderCompleted = true;
                    } else if (++checkCount >= maxChecks) {
                        console.warn('超时：图表可能为空');
                        clearInterval(intervalId);
                        window.renderCompleted = false;
                    }
                } catch(e) {
                    console.error('获取像素数据失败:', e);
                    if (++checkCount >= maxChecks) {
                        clearInterval(intervalId);
                        window.renderCompleted = false;
                    }
                }
            }, checkInterval);
        });
    </script>
</body>
</html>
    `;
}

async function renderEChartsToPNG(jsFilePath, pngOutputPath) {
    const jsContent = fs.readFileSync(jsFilePath, 'utf-8');

    if (hasAsyncRequest(jsContent)) {
        console.log(`跳过 ${jsFilePath} (包含异步请求)`);
        return false;
    }

    const outputDir = path.dirname(pngOutputPath);
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const htmlContent = buildHTML(jsContent);

    const browser = await puppeteer.launch({
        executablePath: CHROME_PATH,
        headless: 'new',
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            '--allow-file-access-from-files'
        ]
    });

    const page = await browser.newPage();
    const logs = [];
    page.on('console', msg => {
        const text = msg.text();
        logs.push(`[${msg.type()}] ${text}`);
        if (msg.type() === 'error') {
            console.error(`PAGE ${msg.type()}:`, text);
        } else if (msg.type() === 'warning') {
            console.warn(`PAGE ${msg.type()}:`, text);
        } else {
            console.log(`PAGE LOG:`, text);
        }
    });

    try {
        await page.setContent(htmlContent, { waitUntil: 'networkidle0', timeout: 30000 });

        // 等待渲染完成标志，最多 30 秒
        const success = await page.waitForFunction(
            'typeof window.renderCompleted !== "undefined"',
            { timeout: 30000 }
        ).then(() => page.evaluate('window.renderCompleted')).catch(() => false);

        if (!success) {
            console.warn(`警告: ${jsFilePath} 渲染可能为空`);
        }

        // 截图
        await page.screenshot({
            path: pngOutputPath,
            clip: { x: 0, y: 0, width: 800, height: 600 }
        });

        // 保存日志
        const logPath = pngOutputPath.replace(/\.png$/, '.log');
        fs.writeFileSync(logPath, logs.join('\n'));

        console.log(`渲染完成: ${pngOutputPath}`);
        return true;
    } catch (err) {
        console.error(`渲染失败 ${jsFilePath}:`, err.message);
        try {
            await page.screenshot({ path: pngOutputPath, clip: { x: 0, y: 0, width: 800, height: 600 } });
        } catch(e) {}
        const logPath = pngOutputPath.replace(/\.png$/, '.log');
        fs.writeFileSync(logPath, logs.join('\n'));
        return false;
    } finally {
        await browser.close();
    }
}

if (require.main === module) {
    const args = process.argv.slice(2);
    if (args.length < 2) {
        console.error('用法: node render_echarts.js <input_js> <output_png>');
        process.exit(1);
    }
    const inputJs = args[0];
    const outputPng = args[1];
    renderEChartsToPNG(inputJs, outputPng).then(success => {
        if (!success) process.exit(1);
    });
}

module.exports = renderEChartsToPNG;