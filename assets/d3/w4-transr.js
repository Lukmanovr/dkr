(function () {
  'use strict';
  const U=window.DKR, $=id=>document.getElementById(id);
  function render(){
    const P=U.pal(), a=+$('w4-tr-scale').value;
    $('w4-tr-value').textContent=a.toFixed(1);
    const svg=U.svgIn('w4-tr-svg',760,380).attr('role','img').attr('aria-label',`Entity space and projected relation space. Two distinct tails have projected distance ${a} to the predicted tail.`);
    const label=(x,y,s)=>svg.append('text').attr('x',x).attr('y',y).attr('text-anchor','middle').attr('font-size',28).attr('fill',P.text).text(s);
    const line=(x,y,X,Y,dash=false)=>svg.append('line').attr('x1',x).attr('y1',y).attr('x2',X).attr('y2',Y).attr('stroke',P.muted).attr('stroke-width',3).attr('stroke-dasharray',dash?'7,6':null);
    const point=(x,y,name,color,dx,dy)=>{svg.append('circle').attr('cx',x).attr('cy',y).attr('r',12).attr('fill',color);label(x+dx,y+dy,name);};
    label(180,40,'Entity space');label(575,40,'Relation space');
    line(380,65,380,350,true);
    for(const ox of [85,480]){line(ox-30,215,ox+225,215);line(ox,335,ox,85);}
    point(85,215,'h',P.yellow,-26,-22);point(265,125,'t₁',P.accent,32,-15);point(265,305,'t₂',P.green,32,40);
    point(480,215,'Mh',P.yellow,0,-27);
    svg.append('defs').append('marker').attr('id','w4-tr-arrow').attr('viewBox','0 0 10 10')
      .attr('refX',9).attr('refY',5).attr('markerWidth',4).attr('markerHeight',4).attr('orient','auto')
      .append('path').attr('d','M0,0 L10,5 L0,10 Z').attr('fill',P.blue);
    line(498,215,632,215).attr('stroke',P.blue).attr('stroke-width',4).attr('marker-end','url(#w4-tr-arrow)');label(570,194,'r');
    svg.append('circle').attr('cx',660).attr('cy',215).attr('r',23).attr('fill','none').attr('stroke',P.blue).attr('stroke-width',4);
    line(660,215,660,215-90*a,true);line(660,215,660,215+90*a,true);
    if(a===0){point(660,215,'Mt₁ = Mt₂',P.green,0,65);}
    else {point(660,215-90*a,'Mt₁',P.accent,0,-28);point(660,215+90*a,'Mt₂',P.green,0,48);}
    const out=$('w4-tr-result');out.dataset.scores=JSON.stringify([-a,-a,-3]);out.dataset.scale=a;
    out.innerHTML=`<div class="w4-grid"><div><strong>Stored tails stay fixed</strong><p>t₁ = (2, 1), t₂ = (2, −1)</p><p>They are two units apart.</p></div><div><strong>Projected tails change</strong><p>Mt₁ = (2, ${a.toFixed(1)}), Mt₂ = (2, ${(-a).toFixed(1)})</p><p>Scores: <strong>${(-a).toFixed(1)} and ${(-a).toFixed(1)}</strong></p></div></div><p>A decoy at (−1, 0) still has score <strong>−3.0</strong>. ${a===0?'Only the relation view merges the true tails. The entity table keeps them distinct.':'Both valid tails miss the landing point. Reduce the vertical scale to reduce both residuals.'}</p>`;
  }
  $('w4-tr-scale').addEventListener('input',render);
  $('w4-tr-reset').addEventListener('click',()=>{$('w4-tr-scale').value=1;render();});
  U.onThemeChange(render);U.lazyBoot('w4-tr-svg',render);
})();
