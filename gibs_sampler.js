// E-LOTUS real-data sampler -- run in a browser console on https://gibs.earthdata.nasa.gov/
// Propagates the real PHISAT-2 TLE with SGP4 (satellite.js), and for every 30-s sub-satellite point
// cuts a 32x32-pixel (~313 km) block from NASA GIBS global daily mosaics (4096x2048, EPSG:4326):
//   optical quick-look   VIIRS_NOAA20_CorrectedReflectance_TrueColor   -> Shannon entropy of luminance
//   thermal-IR quick-look VIIRS_NOAA20_Brightness_Temp_BandI5_Night     -> Shannon entropy of luminance
//   fire detections       VIIRS_NOAA20_Thermal_Anomalies_375m_All       -> number of fire pixels
//   cloud fraction        MODIS_Aqua_Cloud_Fraction_Day / _Night        -> % (decoded with the GIBS colormap)
//   land fraction         OSM_Land_Water_Map                            -> %
// Output: elotus_gibs_30d.txt (binary; 4096-byte JSON header, then lat f32[N], lon f32[N],
//         H_tc u8[N] (x30), H_bt u8[N] (x30), land% u8[N], fire_px u8[N], cloud_day% u8[N],
//         cloud_night% u8[N] (255 = no data), mean_tc u8[N]).  Rename to data/elotus_gibs_30d.bin.
window.__E = {status:'starting', day:-1, log:[], t0:Date.now()};
(async () => {
const E = window.__E;
try {
const S = await import('https://cdn.jsdelivr.net/npm/satellite.js@5.0.0/+esm');
const L1='1 60470U 24149C   26269.50436576  .00007554  00000+0  24499-3 0  9999';
const L2='2 60470  97.3702 348.2033 0000977 167.3518 192.7749 15.32264590117504';
const rec = S.twoline2satrec(L1, L2);
const W=4096, H=2048, CW=1024, CH=512, DAYS=30, STEP=30, PER=86400/STEP, N=DAYS*PER;
const T0 = Date.UTC(2026,7,26,0,0,0);
const lat=new Float32Array(N), lon=new Float32Array(N), px=new Uint16Array(N), py=new Uint16Array(N);
for (let i=0;i<N;i++){ const d=new Date(T0+i*STEP*1000); const pv=S.propagate(rec,d); const g=S.eciToGeodetic(pv.position, S.gstime(d));
  lat[i]=S.degreesLat(g.latitude); lon[i]=S.degreesLong(g.longitude);
  px[i]=Math.min(W-1,Math.max(0,Math.floor((lon[i]+180)/360*W))); py[i]=Math.min(H-1,Math.max(0,Math.floor((90-lat[i])/180*H))); }
E.log.push('track done');
const cmapTxt = await (await fetch('/colormaps/v1.3/MODIS_Cloud_Fraction.xml')).text();
const pal = [...cmapTxt.matchAll(/<ColorMapEntry([^>]*)\/>/g)].map(m=>m[1]).map(e=>({rgb:(e.match(/rgb="([^"]+)"/)||[])[1].split(',').map(Number), nd:/nodata="true"/.test(e), v:Number((e.match(/value="([^"]+)"/)||[])[1])})).filter(p=>!p.nd);
const exact=new Map(pal.map(p=>[p.rgb.join(','),p.v]));
function cfval(r,g,b){ const k=r+','+g+','+b; if(exact.has(k)) return exact.get(k); let best=1e9,bv=255; for(const p of pal){const d=(p.rgb[0]-r)**2+(p.rgb[1]-g)**2+(p.rgb[2]-b)**2; if(d<best){best=d;bv=p.v;}} return best<300?bv:255; }
async function grab(layer, date, w, h, fmt){
  const url=`/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS=${layer}&CRS=EPSG:4326&BBOX=-90,-180,90,180&WIDTH=${w}&HEIGHT=${h}&FORMAT=${fmt}${date?'&TIME='+date:''}`;
  for (let a=0;a<3;a++){ try { const r=await fetch(url); if(!r.ok) throw new Error('HTTP '+r.status); const b=await r.blob(); const bm=await createImageBitmap(b);
    const c=new OffscreenCanvas(w,h); const x=c.getContext('2d',{willReadFrequently:true}); x.drawImage(bm,0,0); return x.getImageData(0,0,w,h).data; } catch(e){ E.log.push(layer+' '+date+' retry '+e); await new Promise(r=>setTimeout(r,3000)); } }
  throw new Error('failed '+layer+' '+date);
}
function lum(d){ const o=new Uint8Array(d.length/4); for(let i=0,j=0;i<d.length;i+=4,j++) o[j]=(d[i]*77+d[i+1]*150+d[i+2]*29)>>8; return o; }
function alphaMask(d){ const o=new Uint8Array(d.length/4); for(let i=0,j=0;i<d.length;i+=4,j++) o[j]=(d[i+3]>0 && (d[i]+d[i+1]+d[i+2])>0)?1:0; return o; }
function cfGrid(d){ const o=new Uint8Array(d.length/4); for(let i=0,j=0;i<d.length;i+=4,j++) o[j]= d[i+3]===0?255:cfval(d[i],d[i+1],d[i+2]); return o; }
const landImg = await grab('OSM_Land_Water_Map', null, W, H, 'image/png');
const land = new Uint8Array(W*H); for(let i=0,j=0;i<landImg.length;i+=4,j++) land[j]= landImg[i]<100?1:0;
const Htc=new Uint8Array(N), Hbt=new Uint8Array(N), LND=new Uint8Array(N), FIRE=new Uint8Array(N), CFD=new Uint8Array(N), CFN=new Uint8Array(N), MTC=new Uint8Array(N);
const hist=new Uint32Array(256);
function entropyAt(img,x,y){ hist.fill(0); const y0=Math.min(H-32,Math.max(0,y-16));
  for(let yy=y0;yy<y0+32;yy++){ const row=yy*W; for(let k=-16;k<16;k++){ const xx=(x+k+W)%W; hist[img[row+xx]]++; } }
  let h=0; for(let v=0;v<256;v++){ if(hist[v]){ const p=hist[v]/1024; h-=p*Math.log2(p);} } return h; }
function meanAt(img,x,y){ const y0=Math.min(H-32,Math.max(0,y-16)); let s=0; for(let yy=y0;yy<y0+32;yy++){const row=yy*W; for(let k=-16;k<16;k++) s+=img[row+(x+k+W)%W];} return s/1024; }
function countAt(m,x,y){ const y0=Math.min(H-32,Math.max(0,y-16)); let s=0; for(let yy=y0;yy<y0+32;yy++){const row=yy*W; for(let k=-16;k<16;k++) s+=m[row+(x+k+W)%W];} return s; }
function cfAt(g,x,y){ const cx=Math.floor(x/4), cy=Math.floor(y/4); const y0=Math.min(CH-8,Math.max(0,cy-4)); let s=0,n=0;
  for(let yy=y0;yy<y0+8;yy++){ for(let k=-4;k<4;k++){ const v=g[yy*CW+(cx+k+CW)%CW]; if(v!==255){s+=v;n++;} } } return n? Math.round(s/n):255; }
const dates=[]; for(let d=0; d<DAYS; d++){ dates.push(new Date(T0+d*86400000).toISOString().slice(0,10)); }
function save(tag){ const hdr={source:'NASA GIBS WMS (epsg4326/best)', layers:{optical:'VIIRS_NOAA20_CorrectedReflectance_TrueColor', tir:'VIIRS_NOAA20_Brightness_Temp_BandI5_Night', fires:'VIIRS_NOAA20_Thermal_Anomalies_375m_All', cloud_day:'MODIS_Aqua_Cloud_Fraction_Day', cloud_night:'MODIS_Aqua_Cloud_Fraction_Night', land:'OSM_Land_Water_Map'}, tle:[L1,L2], t0:'2026-08-26T00:00:00Z', step_s:STEP, n:N, days_done:E.day+1, grid:[W,H], block_px:32, dates, fields:['lat_f32','lon_f32','H_tc_x30_u8','H_bt_x30_u8','land_pct_u8','fire_px_u8','cloud_day_pct_u8(255=nodata)','cloud_night_pct_u8(255=nodata)','mean_tc_u8']};
  const parts=[JSON.stringify(hdr).padEnd(4096,' '), lat.buffer, lon.buffer, Htc, Hbt, LND, FIRE, CFD, CFN, MTC];
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob(parts, {type:'text/plain'})); a.download=`elotus_gibs_${tag}.txt`; document.body.appendChild(a); a.click(); E.log.push('saved '+tag); }
for (let d=0; d<DAYS; d++){
  E.day=d; E.status='day '+d+' '+dates[d];
  const tc=lum(await grab('VIIRS_NOAA20_CorrectedReflectance_TrueColor', dates[d], W, H, 'image/jpeg'));
  const bt=lum(await grab('VIIRS_NOAA20_Brightness_Temp_BandI5_Night', dates[d], W, H, 'image/png'));
  const fire=alphaMask(await grab('VIIRS_NOAA20_Thermal_Anomalies_375m_All', dates[d], W, H, 'image/png'));
  const cfd=cfGrid(await grab('MODIS_Aqua_Cloud_Fraction_Day', dates[d], CW, CH, 'image/png'));
  const cfn=cfGrid(await grab('MODIS_Aqua_Cloud_Fraction_Night', dates[d], CW, CH, 'image/png'));
  for (let i=d*PER;i<(d+1)*PER;i++){ const x=px[i], y=py[i];
    Htc[i]=Math.round(entropyAt(tc,x,y)*30); Hbt[i]=Math.round(entropyAt(bt,x,y)*30);
    LND[i]=Math.round(countAt(land,x,y)/1024*100); FIRE[i]=Math.min(255,countAt(fire,x,y));
    CFD[i]=cfAt(cfd,x,y); CFN[i]=cfAt(cfn,x,y); MTC[i]=Math.round(meanAt(tc,x,y)); }
  if (d===9 || d===19) save('partial_d'+(d+1));
}
save('30d'); E.status='done'; E.elapsed_s=(Date.now()-E.t0)/1000;
} catch(e){ E.status='error'; E.err=String(e)+' '+(e.stack||''); }
})();
