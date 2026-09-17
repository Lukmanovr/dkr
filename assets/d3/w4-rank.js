(function(){
  'use strict';const U=window.DKR;
  const entities={Tatarstan:4,Russia:4.3,Volga:6,Kazan:1,IU:.4};
  const examples=[
    {target:'Tatarstan',known:['Russia','Tatarstan'],candidates:Object.entries(entities).map(([name,x])=>({name,score:-Math.abs(6-1.8-x)}))},
    {target:'ID 1',known:['ID 0','ID 1'],candidates:[9,8,8,1].map((score,i)=>({name:`ID ${i}`,score}))}
  ];
  function render(){
    const P=U.pal(),ex=examples[+document.getElementById('w4-rk-example').value],filtered=document.getElementById('w4-rk-filter').checked;
    const rows=[...ex.candidates].sort((a,b)=>b.score-a.score);
    const eligible=rows.filter(c=>!filtered||!ex.known.includes(c.name)||c.name===ex.target);
    const targetScore=rows.find(c=>c.name===ex.target).score;
    const greater=eligible.filter(c=>c.score>targetScore).length,equal=eligible.filter(c=>c.score===targetScore).length;
    const rank=1+greater+(equal-1)/2;
    const svg=U.svgIn('w4-rk-svg',760,150).attr('role','img').attr('aria-label',`Target ${ex.target}; ${greater} better; ${equal} tied including target; rank ${rank}`);
    const vals=[['Better',greater],['Tied',equal],['Rank',rank]];
    vals.forEach(([label,value],i)=>{
      svg.append('text').attr('x',130+i*250).attr('y',42).attr('text-anchor','middle').attr('font-size',25).attr('fill',P.text).text(label);
      svg.append('text').attr('x',130+i*250).attr('y',103).attr('text-anchor','middle').attr('font-size',43).attr('font-weight',700).attr('fill',P.text).text(value);
      svg.append('line').attr('x1',40+i*250).attr('x2',220+i*250).attr('y1',120).attr('y2',120).attr('stroke',P.border);
    });
    const result=document.getElementById('w4-rk-result');result.dataset.rank=rank;
    result.innerHTML=`<p><strong>Target: ${ex.target}</strong> · ${filtered?'filtered':'raw'} · tied count includes the target</p><table><thead><tr><th>Tail / ID</th><th>Score</th><th>Action</th></tr></thead><tbody>`+rows.map(c=>{
      const target=c.name===ex.target,excluded=!eligible.includes(c);
      return `<tr class="${target?'w4-target':excluded?'w4-excluded':''}"><td>${c.name}</td><td>${c.score.toFixed(1)}</td><td>${target?'Target: keep':excluded?'Known true: exclude':ex.known.includes(c.name)?'Known true: still counted':'Keep'}</td></tr>`;
    }).join('')+`</tbody></table><p class="w4-readout">Rank = 1 + ${greater} + (${equal} − 1)/2 = <strong>${rank}</strong><br>Reciprocal rank = ${(1/rank).toFixed(3)} · Hits@1 = ${rank<=1?1:0}</p>`;
  }
  ['w4-rk-example','w4-rk-filter'].forEach(id=>document.getElementById(id).addEventListener('change',render));
  U.onThemeChange(render);U.lazyBoot('w4-rk-svg',render);
})();
