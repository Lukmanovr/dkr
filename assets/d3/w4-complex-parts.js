(function () {
  'use strict';
  const U=window.DKR, $=id=>document.getElementById(id);
  function render() {
    const P=U.pal(), c=Number($('w4-cp-real').value), d=Number($('w4-cp-imag').value);
    const S=5*c, A=-5*d, forward=S+A, reverse=S-A;
    const fmt=n=>String(Object.is(n,-0)?0:n).replace('-', '−');
    $('w4-cp-c').textContent=fmt(c); $('w4-cp-d').textContent=fmt(d);
    const svg=U.svgIn('w4-cp-svg',760,355).attr('role','img')
      .attr('aria-label',`Shared contribution ${S}, directional contribution ${A}; forward ${forward}, reverse ${reverse}.`);
    const label=(x,y,value,size=28)=>svg.append('text').attr('x',x).attr('y',y).attr('text-anchor','middle')
      .attr('font-size',size).attr('fill',P.text).text(value);
    for(const [cx,y,title,value,color] of [[195,70,'Shared S',S,P.blue],[565,70,'Directional A',A,P.accent],
                                         [195,250,'Forward: S + A',forward,P.blue],[565,250,'Reverse: S − A',reverse,P.accent]]) {
      label(cx,y-30,title);
      svg.append('rect').attr('x',cx-140).attr('y',y-8).attr('width',280).attr('height',84)
        .attr('rx',9).attr('fill',P.paper).attr('stroke',color).attr('stroke-width',3);
      label(cx,y+50,fmt(value),43);
    }
    label(380,185,'Swap the endpoints: keep S, flip A',27);
    const out=$('w4-cp-result');
    Object.assign(out.dataset,{shared:S,directional:A,forward,reverse});
    out.innerHTML=`<p><strong>h = 1 + 2i; t = 3 + i; r = ${fmt(c)} ${d<0?'−':'+'} ${Math.abs(d)}i.</strong></p>`+
      `<p>S = c(1×3 + 2×1) = ${fmt(S)}. A = d(1×1 − 2×3) = ${fmt(A)}.</p>`+
      `<p>Forward − reverse = 2A = <strong>${fmt(2*A)}</strong>. ${d===0?'The real relation gives equal scores.':c===0?'The purely imaginary relation gives opposite scores.':'The relation combines a shared contribution and a directional contribution.'}</p>`;
  }
  for(const id of ['w4-cp-real','w4-cp-imag']) $(id).addEventListener('input',render);
  $('w4-cp-conjugate').addEventListener('click',()=>{$('w4-cp-imag').value=-Number($('w4-cp-imag').value);render();});
  $('w4-cp-reset').addEventListener('click',()=>{$('w4-cp-real').value=1;$('w4-cp-imag').value=0;render();});
  U.onThemeChange(render); U.lazyBoot('w4-cp-svg',render);
})();
