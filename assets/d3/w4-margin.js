(function () {
  'use strict';
  const U=window.DKR,$=id=>document.getElementById(id);
  function render(){
    const P=U.pal(),n=+$('w4-ma-negative').value,g=+$('w4-ma-gamma').value,loss=Math.max(0,g+1-n),sx=x=>220+110*x;
    $('w4-ma-value').textContent=n.toFixed(1);$('w4-ma-gvalue').textContent=g.toFixed(1);
    const svg=U.svgIn('w4-ma-svg',760,280).attr('role','img').attr('aria-label',`Positive distance 1, negative distance ${n}, required distance ${1+g}, margin loss ${loss}.`);
    const label=(x,y,s)=>svg.append('text').attr('x',x).attr('y',y).attr('text-anchor','middle').attr('font-size',28).attr('fill',P.text).text(s);
    for(const [y,v,name,color] of [[90,1,'Positive',P.green],[180,n,'Negative',P.accent]]){
      label(105,y+10,name);svg.append('rect').attr('x',220).attr('y',y-17).attr('width',110*v).attr('height',34).attr('rx',5).attr('fill',color);
      label(sx(v)+40,y+10,v.toFixed(1));
    }
    svg.append('line').attr('x1',sx(1+g)).attr('x2',sx(1+g)).attr('y1',50).attr('y2',215).attr('stroke',P.blue).attr('stroke-width',3).attr('stroke-dasharray','7,6');
    label(sx(1+g),30,'1 + γ');label(435,261,'Distance → larger is farther');
    const out=$('w4-ma-result');out.dataset.loss=loss;
    out.innerHTML=`<p>Scores: positive <strong>−1.0</strong>, negative <strong>${(-n).toFixed(1)}</strong>.</p><p>Loss = max(0, ${g.toFixed(1)} + 1.0 − ${n.toFixed(1)}) = <strong>${loss.toFixed(1)}</strong>.</p><p>${loss>0?'The hinge is active: reducing positive distance or increasing negative distance reduces this loss. Shared embeddings couple those changes.':'The sampled pair meets the margin. Its hinge contribution is zero; other sampled pairs can still cause updates.'}</p>`;
  }
  ['w4-ma-negative','w4-ma-gamma'].forEach(id=>$(id).addEventListener('input',render));
  U.onThemeChange(render);U.lazyBoot('w4-ma-svg',render);
})();
