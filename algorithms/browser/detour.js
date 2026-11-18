// Recommendation rule extracted from the Q-RITE viewer index.html.
// budget is a fraction (0.1 = 10%); use 1 for the viewer's unlimited option.
function pickBest(cs, budget){
  const b0=cs.find(c=>c.kind==='straight');
  const L0=b0?b0.length_km:cs[0].length_km;
  const maxLength=budget>=1?Infinity:L0*(1+budget)+1e-9;
  cs.forEach(c=>{
    c._over = c.kind!=='straight' && c.length_km>maxLength;
    c._best = false;
  });
  const ok=cs.filter(c=>c.kind!=='straight'&&!c._over);
  const best=ok.slice().sort((a,b)=>a.cavity_km-b.cavity_km||a.length_km-b.length_km)[0];
  if(best)best._best=true;
  return {L0,best};
}


if (typeof module !== 'undefined') module.exports = { pickBest };
