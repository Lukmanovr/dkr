/* One deterministic forward pass; arithmetic stays visible in HTML at mobile sizes. */
(function () {
  'use strict';
  const U = window.DKR;
  const tail = document.getElementById('w4-te-tail');
  const ry = document.getElementById('w4-te-y');
  function render() {
    const P = U.pal(), y = Number(ry.value), ty = tail.value === '0' ? 2 : 1;
    const predicted = [3, 1+y], residual = [0, 1+y-ty];
    const distance = Math.hypot(...residual);
    const svg = U.svgIn('w4-te-svg', 620, 470).attr('role', 'img')
      .attr('aria-label', `Head plus relation predicts (3, ${predicted[1].toFixed(1)}); Innopolis is at (3,2), Kazan at (3,1). Selected tail distance ${distance.toFixed(2)}`);
    // Equal scales: Euclidean lengths and the grid tell the same geometric story.
    const sx = x => 90 + x*110, sy = y => 390-y*110;
    for (let x=0; x<=4; x++) {
      svg.append('line').attr('x1',sx(x)).attr('x2',sx(x)).attr('y1',35).attr('y2',395).attr('stroke',P.border);
      svg.append('text').attr('x',sx(x)).attr('y',432).attr('text-anchor','middle').attr('font-size',26).attr('fill',P.muted).text(x);
    }
    for (let y=0; y<=3; y++) {
      svg.append('line').attr('x1',sx(0)).attr('x2',sx(4)).attr('y1',sy(y)).attr('y2',sy(y)).attr('stroke',P.border);
      svg.append('text').attr('x',sx(0)-40).attr('y',sy(y)+9).attr('text-anchor','middle').attr('font-size',26).attr('fill',P.muted).text(y);
    }
    svg.append('defs').append('marker').attr('id','w4-te-arrow').attr('viewBox','0 0 10 10')
      .attr('refX',9).attr('refY',5).attr('markerWidth',4).attr('markerHeight',4).attr('orient','auto')
      .append('path').attr('d','M0,0 L10,5 L0,10 Z').attr('fill',P.blue);
    const length = Math.hypot(2,y);
    // Stop outside the landing ring so the arrowhead stays visible at exact fits.
    svg.append('line').attr('x1',sx(1)).attr('y1',sy(1)).attr('x2',sx(3)-28*2/length).attr('y2',sy(1+y)+28*y/length)
      .attr('stroke',P.blue).attr('stroke-width',5).attr('marker-end','url(#w4-te-arrow)');
    svg.append('line').attr('x1',sx(3)).attr('y1',sy(1+y)).attr('x2',sx(3)).attr('y2',sy(ty))
      .attr('stroke',P.accent).attr('stroke-width',5).attr('stroke-dasharray','7,7');
    [[1,1,'h',P.yellow],[3,2,'B',P.green],[3,1,'C',P.accent]].forEach(([x,y,label,color])=>{
      svg.append('circle').attr('cx',sx(x)).attr('cy',sy(y)).attr('r',17).attr('fill',color);
      svg.append('text').attr('x',sx(x)+31).attr('y',sy(y)+9).attr('font-size',28).attr('fill',P.text).text(label);
    });
    svg.append('circle').attr('cx',sx(3)).attr('cy',sy(1+y)).attr('r',23).attr('fill','none').attr('stroke',P.blue).attr('stroke-width',3);
    document.getElementById('w4-te-yval').textContent = y.toFixed(1);
    const result = document.getElementById('w4-te-result');
    result.dataset.score = String(-distance);
    result.innerHTML = `<p><strong>B · Innopolis:</strong> score ${(-Math.abs(y-1)).toFixed(2)} &nbsp; <strong>C · Kazan:</strong> score ${(-Math.abs(y)).toFixed(2)}</p><p>Prediction: (1, 1) + (2, ${y.toFixed(1)}) = <strong>(3, ${predicted[1].toFixed(1)})</strong>.</p><p>Selected tail: residual (0, ${residual[1].toFixed(1)}) → distance ${distance.toFixed(2)} → <strong>score ${(-distance).toFixed(2)}</strong>.</p>`;
  }
  tail.addEventListener('change',render); ry.addEventListener('input',render);
  document.getElementById('w4-te-reset').addEventListener('click',()=>{tail.value='0';ry.value='1';render();});
  U.onThemeChange(render); U.lazyBoot('w4-te-svg',render);
})();
