(function () {
  'use strict';
  const U=window.DKR;
  // Each cell states the algebra AND its scope; no universal capability checkmarks.
  const cases={
    symmetry: {
      TransR:['Coincide only after projection','r = 0; Mᵣa = Mᵣb, while a ≠ b.','A relation-specific kernel permits distinct stored entities to have identical projected vectors.'],
      TransE:['Exact fits collapse','a + r = b; b + r = a ⇒ r = 0 and a = b.','This assumes both distances are zero. Approximate symmetric rankings remain possible.'],
      DistMult:['Equal scores are automatic','Σ aᵢrᵢbᵢ = Σ bᵢrᵢaᵢ.','Every parameter setting is symmetric, including relations that should be directed.'],
      ComplEx:['Choose a real relation','Im(r) = 0 ⇒ f(a,r,b) = f(b,r,a).','With complex entities, imaginary relation components can also permit direction.'],
      RotatE:['A half-turn exchanges points','a = 1; r = −1; b = −1. Then ar = b and br = a.','Distinct entity coordinates can have zero distance in both directions.']
    },
    direction: {
      TransR:['Translate in the relation space','Mᵣa + r = Mᵣb with r ≠ 0 gives a nonzero reverse residual 2r.','A sufficient exact-fit construction; the projection must retain the distinction.'],
      TransE:['A translation has direction','a = 0; r = 1; b = 1. Forward score 0; reverse score −2.','This example establishes possibility, not that training will discover it.'],
      DistMult:['Cannot distinguish a reversed triple','f(a,r,b) = f(b,r,a), for every embedding.','An optimizer cannot remove an identity built into the scorer.'],
      ComplEx:['The conjugate permits direction','a = 1+i; r = i; b = −1+i. Scores: 2 and −2.','The tail conjugate and real-part operation both matter.'],
      RotatE:['A quarter-turn has direction','a = 1+i; r = i; b = −1+i. Scores: 0 and −2√2.','A unit relation rotates; the final distance makes a real score.']
    },
    inversion: {
      TransR:['Share a projection; negate the shift','M₂ = M₁ and r₂ = −r₁ undo the projected translation.','These are sufficient conditions, not constraints imposed by the default optimizer.'],
      TransE:['Undo the translation','r₂ = −r₁: a+r₁=b ⇔ b+r₂=a.','An exact transformation identity.'],
      DistMult:['Reverse scores are already equal','Using the same vector gives f(a,r₁,b)=f(b,r₂,a).','It cannot model distinct forward/reverse directionality.'],
      ComplEx:['Conjugate the relation','r₂ = conjugate(r₁) gives f(a,r₁,b)=f(b,r₂,a).','This is a score identity, with no requirement for zero distance.'],
      RotatE:['Negate the angles','r₂ = conjugate(r₁) = 1/r₁ for unit rotations.','Rotating forward and then backward returns the same point.']
    },
    composition: {
      TransR:['Compose offsets and kernel directions','Choose M₃ to annihilate ker(M₁) + ker(M₂), and r₃ = M₃(g₁ + g₂).','This exact construction can discard distinctions. It is not ordinary multiplication of arbitrary learned projections.'],
      TransE:['Add displacements','(a+r₁)+r₂ = a+(r₁+r₂).','Applies to exact transformations; high scores alone do not prove a rule.'],
      DistMult:['Bilinear scores are not point equalities','Diagonal relation matrices can be multiplied.','This can support rule mining, but multiplying scores is not a general entailment guarantee.'],
      ComplEx:['Separate score products from paths','A high score through b does not assert ar₁=b.','Complex relation products can be studied; no general path rule follows from this score alone.'],
      RotatE:['Multiply rotations; add angles','(a ∘ r₁) ∘ r₂ = a ∘ (r₁ ∘ r₂).','TransE also composes. This property is not unique to rotations.']
    },
    many: {
      TransR:['Distinct entities, identical projections','Mᵣt₁ = Mᵣt₂ = Mᵣh + r, while t₁ ≠ t₂.','A nontrivial kernel can hide the tails’ difference for this relation without deleting it from the entity table.'],
      TransE:['Zero-distance tails coincide','h+r=t₁ and h+r=t₂ ⇒ t₁=t₂.','Distinct tails may still have small nonzero distances and good ranks.'],
      DistMult:['Several tails can align well','Multiple tails can have a large dot product with h ∘ r.','High scores do not require their vectors to be identical.'],
      ComplEx:['Several tails can align well','Several conjugated tails can give large real products.','Check ranking quality empirically; algebra alone does not select a winner.'],
      RotatE:['One exact landing point','h ∘ r=t₁ and h ∘ r=t₂ ⇒ t₁=t₂.','Nearby distinct tails can both score well; exact-fit collapse is a narrower claim.']
    }
  };
  function render(){
    const P=U.pal(), pat=document.getElementById('w4-pt-pattern').value;
    const model=document.getElementById('w4-pt-model').value;
    const [title,algebra,scope]=cases[pat][model];
    const svg=U.svgIn('w4-pt-svg',760,210).attr('role','img').attr('aria-label',`Schematic relation pattern: ${pat}. Read the model-specific algebra below.`);
    svg.append('defs').append('marker').attr('id','w4-pattern-arrow').attr('markerWidth',7).attr('markerHeight',7).attr('refX',6).attr('refY',3).attr('orient','auto')
      .append('path').attr('d','M0,0 L6,3 L0,6').attr('fill',P.muted);
    const node=(x,y,label,color)=>{
      svg.append('circle').attr('cx',x).attr('cy',y).attr('r',24).attr('fill',color);
      svg.append('text').attr('x',x).attr('y',y+9).attr('text-anchor','middle').attr('font-size',28).attr('fill','#fff').text(label);
    };
    const edge=(x1,y1,x2,y2,label,lx,ly)=>{
      svg.append('line').attr('x1',x1).attr('x2',x2).attr('y1',y1).attr('y2',y2).attr('stroke',P.muted).attr('stroke-width',3).attr('marker-end','url(#w4-pattern-arrow)');
      svg.append('text').attr('x',lx).attr('y',ly).attr('text-anchor','middle').attr('font-size',28).attr('fill',P.text).text(label);
    };
    if(pat==='many'){
      node(170,105,'h',P.yellow);node(590,45,'t₁',P.green);node(590,165,'t₂',P.accent);
      edge(198,101,554,49,'r',380,49);edge(198,109,554,161,'r',380,180);
    }else if(pat==='composition'){
      node(110,105,'a',P.yellow);node(380,105,'b',P.accent);node(650,105,'c',P.green);
      edge(143,105,343,105,'r₁',240,74);edge(413,105,613,105,'r₂',515,74);
    }else{
      node(170,105,'a',P.yellow);node(590,105,'b',P.green);
      edge(200,90,553,90,pat==='inversion'?'r₁':'r',380,62);
      if(pat==='symmetry'||pat==='inversion')edge(560,125,207,125,pat==='inversion'?'inverse r₂':'r',380,169);
      else svg.append('text').attr('x',380).attr('y',165).attr('text-anchor','middle').attr('font-size',28).attr('fill',P.text).text('reverse is a different claim');
    }
    const result=document.getElementById('w4-pt-result');
    result.innerHTML=`<p class="w4-eyebrow">Schematic pattern above · ${model} below</p><p><strong>${title}</strong></p><p>${algebra}</p><p><strong>Scope:</strong> ${scope}</p>`;
  }
  ['w4-pt-pattern','w4-pt-model'].forEach(id=>document.getElementById(id).addEventListener('change',render));
  U.onThemeChange(render); U.lazyBoot('w4-pt-svg',render);
})();
