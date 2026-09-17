(function () {
  'use strict';
  const U=window.DKR,$=id=>document.getElementById(id);
  function render(){
    const P=U.pal(),deg=+$('w4-ro-angle').value,a=deg*Math.PI/180,inverse=$('w4-ro-inverse').checked;
    const t=[Math.cos(a),Math.sin(a)],back=inverse?[1,0]:[Math.cos(2*a),Math.sin(2*a)],d=Math.hypot(back[0]-1,back[1]);
    const fmt=v=>(Math.abs(v)<1e-8?0:v).toFixed(2),sx=x=>380+145*x,sy=y=>210-145*y;
    $('w4-ro-value').textContent=deg+'°';
    const svg=U.svgIn('w4-ro-svg',760,420).attr('role','img').attr('aria-label',`Unit circle: head at (1,0), tail at angle ${deg} degrees. Return distance ${fmt(d)}.`);
    const defs=svg.append('defs');
    for(const [name,color] of [['forward',P.green],['return',P.accent]])defs.append('marker').attr('id',`w4-ro-${name}-arrow`).attr('markerWidth',7).attr('markerHeight',7).attr('refX',6).attr('refY',3).attr('orient','auto').append('path').attr('d','M0,0 L6,3 L0,6').attr('fill',color);
    const line=(x,y,X,Y,c,dash)=>svg.append('line').attr('x1',x).attr('y1',y).attr('x2',X).attr('y2',Y).attr('stroke',c).attr('stroke-width',3).attr('stroke-dasharray',dash?'7,6':null);
    const label=(x,y,s)=>svg.append('text').attr('x',x).attr('y',y).attr('font-size',28).attr('text-anchor','middle').attr('fill',P.text).text(s);
    line(180,210,600,210,P.muted);line(380,390,380,25,P.muted);
    svg.append('circle').attr('cx',380).attr('cy',210).attr('r',145).attr('fill','none').attr('stroke',P.border).attr('stroke-width',2);
    const arc=(start,turn,radius,name,color)=>{
      if(Math.abs(turn)<1e-8)return;
      const x=angle=>380+radius*Math.cos(angle),y=angle=>210-radius*Math.sin(angle);
      svg.append('path').attr('d',`M${x(start)},${y(start)} A${radius},${radius} 0 0 ${turn>0?0:1} ${x(start+turn)},${y(start+turn)}`).attr('fill','none').attr('stroke',color).attr('stroke-width',3).attr('marker-end',`url(#w4-ro-${name}-arrow)`);
    };
    arc(0,a,112,'forward',P.green);arc(a,inverse?-a:a,170,'return',P.accent);
    line(380,210,sx(t[0]),sy(t[1]),P.green);line(380,210,525,210,P.yellow);
    line(sx(back[0]),sy(back[1]),525,210,P.accent,true);
    svg.append('circle').attr('cx',525).attr('cy',210).attr('r',13).attr('fill',P.yellow);
    svg.append('circle').attr('cx',sx(t[0])).attr('cy',sy(t[1])).attr('r',10).attr('fill',P.green);
    svg.append('circle').attr('cx',sx(back[0])).attr('cy',sy(back[1])).attr('r',23).attr('fill','none').attr('stroke',P.accent).attr('stroke-width',4);
    label(650,220,'real');label(460,32,'imaginary');label(575,252,'h');label(350,252,'0');
    const out=$('w4-ro-result');out.dataset.reverseScore=-d;out.dataset.tail=JSON.stringify(t);out.dataset.back=JSON.stringify(back);
    out.innerHTML=`<p><strong>Gold point:</strong> h = (1, 0). <strong>Green point:</strong> t = (${t.map(fmt).join(', ')}).</p><p>Forward: ${deg}° → score <strong>0</strong>. Return: ${inverse?-deg:deg}° → landing (${back.map(fmt).join(', ')}) → score <strong>${fmt(-d)}</strong>.</p><p>${inverse?'An inverse relation negates the angle; it returns to the head for every slider value.':Math.abs(d)<1e-8?'This angle is self-inverse. At 180°, both directions fit while the two entity points remain distinct.':'The same relation does not undo this rotation. A high forward score need not mean a high reverse score.'}</p>`;
  }
  $('w4-ro-angle').addEventListener('input',render);$('w4-ro-inverse').addEventListener('change',render);
  for(const [id,a] of [['w4-ro-quarter',90],['w4-ro-half',180]])$(id).addEventListener('click',()=>{$('w4-ro-angle').value=a;render();});
  U.onThemeChange(render);U.lazyBoot('w4-ro-svg',render);
})();
