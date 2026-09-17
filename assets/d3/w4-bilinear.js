(function () {
  'use strict';
  const U=window.DKR,$=id=>document.getElementById(id);
  function render(){
    const P=U.pal(), complex=$('w4-bi-model').value==='ComplEx',w=+$('w4-bi-weight').value,swap=$('w4-bi-swap').checked;
    $('w4-bi-value').textContent=w.toFixed(1);$('w4-bi-param').textContent=complex?'Imaginary relation component β':'Relation coordinate r₂';
    const h=complex?(swap?[-1,1]:[1,1]):(swap?[1,2]:[2,1]);
    const t=complex?(swap?[1,1]:[-1,1]):(swap?[2,1]:[1,2]);
    const hr=complex?[-h[1]*w,h[0]*w]:[h[0],h[1]*w];
    const terms=hr.map((v,i)=>v*t[i]),score=terms[0]+terms[1],reverse=complex?-score:score;
    const fmt=v=>Math.abs(v)<1e-9?'0':Number(v.toFixed(2)).toString();
    const svg=U.svgIn('w4-bi-svg',760,300).attr('role','img').attr('aria-label',`${complex?'ComplEx':'DistMult'} coordinate contributions ${terms.join(' and ')} sum to ${score}. Reversed score ${reverse}.`);
    const label=(x,y,s,size=30)=>svg.append('text').attr('x',x).attr('y',y).attr('text-anchor','middle').attr('font-size',size).attr('fill',P.text).text(s);
    [210,550].forEach((x,i)=>{
      svg.append('rect').attr('x',x-130).attr('y',48).attr('width',260).attr('height',84).attr('rx',10).attr('fill',P.paper).attr('stroke',P.blue).attr('stroke-width',2);
      label(x,30,complex?(i?'Imaginary channel':'Real channel'):`Coordinate ${i+1}`,27);
      label(x,103,`${fmt(hr[i])} × ${fmt(t[i])} = ${fmt(terms[i])}`);
      svg.append('line').attr('x1',x).attr('y1',139).attr('x2',380).attr('y2',207).attr('stroke',P.muted).attr('stroke-width',3);
    });
    label(380,251,`Score = ${fmt(score)}`,43);
    const out=$('w4-bi-result');out.dataset.score=score;out.dataset.reverse=reverse;
    out.innerHTML=`<p><strong>${complex?'ComplEx: r = βi':'DistMult: r = (1, r₂)'}</strong></p><p>${complex?'Real/imaginary pairs':'Vectors'}: head (${h.join(', ')}), tail (${t.join(', ')}).</p><p>Transform head → (${hr.map(fmt).join(', ')}). Multiply with tail coordinates → (${terms.map(fmt).join(', ')}). Add → <strong>${fmt(score)}</strong>.</p><p>Swapping this pair back gives <strong>${fmt(reverse)}</strong>. ${complex?'Re(hr × conjugate(t)) equals this two-channel dot product. The imaginary relation can change the sign on reversal.':'The same three real factors are multiplied in a different order, so the score cannot change.'}</p>`;
  }
  $('w4-bi-model').addEventListener('change',()=>{$('w4-bi-weight').value=$('w4-bi-model').value==='ComplEx'?1:3;$('w4-bi-swap').checked=false;render();});
  $('w4-bi-weight').addEventListener('input',render);$('w4-bi-swap').addEventListener('change',render);
  U.onThemeChange(render);U.lazyBoot('w4-bi-svg',render);
})();
