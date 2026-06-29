// 把 markdown-it 渲染出的 ```mermaid 代码块（<pre><code class="language-mermaid">…</code></pre>）
// 转成 mermaid.js 可识别的 <div class="mermaid">…</div>，再交给 mermaid.run() 绘制。
(function () {
  function setupMermaid() {
    if (typeof mermaid === 'undefined') return;

    // 跟随 Fluid 主题的明暗模式
    var isDark = document.documentElement.getAttribute('data-user-color-scheme') === 'dark'
      || (document.body && document.body.classList.contains('dark-mode'));
    mermaid.initialize({
      startOnLoad: false,
      theme: isDark ? 'dark' : 'default',
      securityLevel: 'loose',
      flowchart: { useMaxWidth: true, htmlLabels: true }
    });

    // markdown-it 默认把 ```mermaid 渲染成 <pre><code class=" mermaid">，故用属性匹配兜底
    var blocks = document.querySelectorAll('pre code.language-mermaid, code.language-mermaid, code[class~="mermaid"]');
    blocks.forEach(function (code) {
      var div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = code.textContent;
      var pre = code.closest('pre');
      if (pre && pre.parentNode) {
        pre.parentNode.replaceChild(div, pre);
      } else if (code.parentNode) {
        code.parentNode.replaceChild(div, code);
      }
    });

    if (blocks.length > 0) {
      mermaid.run({ querySelector: '.mermaid' });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupMermaid);
  } else {
    setupMermaid();
  }
})();
