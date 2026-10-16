'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {chromium}=require('/home/zzhang/.local/share/atomx-nist-atlas/browser-check/node_modules/playwright');
const here=__dirname,contract=JSON.parse(fs.readFileSync(path.join(here,'contract.json')));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const overlay=JSON.parse(fs.readFileSync(path.join(contract.site,contract.new_overlay)));
const old=JSON.parse(fs.readFileSync(path.join(here,'old_models.json')));
const checks=[],errors=[],base='http://127.0.0.1:39317',started=new Date().toISOString();
const check=(name,passed)=>checks.push({name,passed:Boolean(passed)});
const fmt=n=>n.toLocaleString('en-US',{maximumFractionDigits:8});
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/home/zzhang/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1080},acceptDownloads:true});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  const response=await page.goto(base+'/#Yb:1:models');await page.waitForSelector('.correlated-summary');
  check('HTTP200_periodic118',response.status()===200&&await page.locator('.element').count()===118);
  check('18models72rows',await page.locator('.model').count()===18&&await page.locator('.model-lines tbody tr').count()===72);
  check('complete17plus65points',await page.locator('.trajectory-points tbody tr').count()===82&&await page.locator('.trajectory-states tbody tr').count()===410);
  const summary=await page.locator('.correlated-summary').innerText();
  check('honest_scope_and_failed_gates',summary.includes('1046 CSF')&&summary.includes('29径向轨道')&&summary.includes('3451')&&summary.includes('均未通过')&&summary.includes('不按实验gap挑中间点'));
  check('shared_start_explicit',summary.includes('累计 80 个更新')&&summary.includes('此前16模型64行保持原值')&&summary.includes('18模型72行'));
  for(const m of overlay.models){
   const block=page.locator('.model[data-model-id="'+m.id+'"]'),text=await block.innerText();
   check(m.id+'/status_and_all4goalsFAIL',text.includes('29轨道MCDHF · 原驻值未过')&&await block.locator('.model-lines .bad').count()===4);
   const rows=await block.locator('.model-lines tbody tr').allTextContents();
   check(m.id+'/four_actual_lines_ASD_gaps',rows.length===4&&m.rows.every((r,i)=>rows[i].includes(fmt(r.calculated_cm1))&&rows[i].includes(fmt(r.reference_cm1))&&rows[i].includes(fmt(r.error_cm1))));
   check(m.id+'/missing_sigma_not_zero',await block.locator('.model-lines tbody tr').evaluateAll(rs=>rs.every(r=>r.cells[5].textContent==='—')));
   check(m.id+'/lower_upper_and_moments',text.includes('上态 ΔHa')&&text.includes('下态 ΔHa')&&text.includes('谱线 Δcm⁻¹')&&text.includes('重心 / 分裂'));
   check(m.id+'/metadata_is_not_replay',text.includes('闭合元数据（不是独立数值重放）')&&text.includes(m.sha256));
   const run=overlay.runs.find(r=>r.model_id===m.id),article=page.locator('[data-run-model="'+m.id+'"]');
   await article.locator('.trajectory-details > summary').click();
   check(m.id+'/full_trajectory_visible',await article.locator('.trajectory-points').isVisible()&&await article.locator('.trajectory-points tbody tr').count()===run.trajectory.length);
   const end=await article.locator('.trajectory-points tbody tr').last().innerText();
   check(m.id+'/terminal_RMS_EOL_residual_visible',end.includes(fmt(m.nominal_gap_RMS_cm1))&&end.includes(fmt(m.EOL_Ha))&&end.includes(m.orbital_residual.toExponential(8)));
  }
  for(const m of old){
   const rows=await page.locator('.model[data-model-id="'+(m.id||m.saved_run)+'"] .model-lines tbody tr').allTextContents();
   check('prior_values/'+(m.id||m.saved_run),rows.length===4&&m.rows.every((r,i)=>rows[i].includes(fmt(r.calculated_cm1))));
  }
  const downloadEvent=page.waitForEvent('download');await page.locator('a[download][href="updates/yb-correlated29-run003-endpoint16.json"]').click();
  const download=await downloadEvent;await download.saveAs(path.join(here,'downloaded_update.json'));
  check('overlay_download_exact',sha(path.join(here,'downloaded_update.json'))===sha(path.join(contract.site,contract.new_overlay)));
  await page.locator('[data-tab="sources"]').click();
  check('new_and_old_source_chains',await page.locator('#content').innerText().then(t=>t.includes('原生29轨道相关MCDHF')&&t.includes('核场与闭核68')&&t.includes('近期 MCDHF')));
  for(const s of overlay.source_files){const res=await context.request.get(base+'/'+s.asset);check('source_download/'+s.asset,res.status()===200&&crypto.createHash('sha256').update(await res.body()).digest('hex')===s.sha256)}
  await page.locator('[data-tab="levels"]').click();check('unchanged349Yb_reference_levels',await page.locator('#page-label').innerText().then(t=>t.includes('349 条')));
  await page.locator('[data-z="10"]').click();await page.waitForFunction(()=>location.hash.startsWith('#Ne:0:'));
  await page.locator('[data-z="70"]').click();await page.locator('#ion').selectOption('1');await page.locator('[data-tab="models"]').click();await page.waitForSelector('.correlated-summary');
  check('cached_revisit18_no_duplicates',await page.locator('.model').count()===18&&await page.locator('.model-lines tbody tr').count()===72);
  await page.locator('.correlated-summary').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(here,'desktop.png')});
  const mobile=await browser.newContext({viewport:{width:390,height:844},isMobile:true,deviceScaleFactor:1});
  const mp=await mobile.newPage();mp.on('pageerror',e=>errors.push(e.message));await mp.goto(base+'/#Yb:1:models');await mp.waitForSelector('.correlated-summary');
  check('mobile18models72rows',await mp.locator('.model').count()===18&&await mp.locator('.model-lines tbody tr').count()===72);
  check('mobile_tables_scrollable',await mp.locator('.correlated-summary .table-wrap').evaluateAll(rs=>rs.every(r=>getComputedStyle(r).overflowX==='auto')));
  await mp.locator('.correlated-summary').scrollIntoViewIfNeeded();await mp.screenshot({path:path.join(here,'mobile.png')});await mobile.close();
  const fallback=await browser.newContext();const fp=await fallback.newPage();fp.on('pageerror',e=>errors.push(e.message));
  await fp.route('**/updates/yb-correlated29-run003-endpoint16.json',r=>r.fulfill({status:503,body:'unavailable'}));await fp.goto(base+'/#Yb:1:models');await fp.waitForSelector('.virtual-summary');
  check('new_overlay_failure_retains16models64rows',await fp.locator('.model').count()===16&&await fp.locator('.model-lines tbody tr').count()===64&&await fp.locator('.correlated-summary').count()===0);
  check('failure_visible_no_new_claim',await fp.locator('#content').innerText().then(t=>t.includes('原生29轨道MCDHF末态读取失败')));
  await fallback.close();await context.close();check('no_JS_errors',errors.length===0);
 }finally{await browser.close()}
 const report={all_passed:checks.every(c=>c.passed),checks_count:checks.length,checks,javascript_errors:errors,started_utc:started,finished_utc:new Date().toISOString(),page_sha256:sha(path.join(contract.site,'dist/index.html')),overlay_sha256:sha(path.join(contract.site,contract.new_overlay)),driver_sha256:sha(__filename),contract_sha256:sha(path.join(here,'contract.json')),browser:'Existing Playwright Chromium1243; desktop1440x1080/mobile390x844',models:18,rows:72,original_models_preserved:16,original_rows_preserved:64,trajectory_rows:82,science_calls:0,publication_performed:false};
 fs.writeFileSync(path.join(here,'browser_result.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify({all_passed:report.all_passed,checks_count:checks.length,failed:checks.filter(c=>!c.passed),javascript_errors:errors}));
 if(!report.all_passed)process.exitCode=1;
})().catch(e=>{fs.writeFileSync(path.join(here,'browser_failure.json'),JSON.stringify({error:e.stack,checks,errors},null,2)+'\n');console.error(e.stack);process.exitCode=1});
