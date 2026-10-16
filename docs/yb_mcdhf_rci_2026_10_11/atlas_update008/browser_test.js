'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {chromium}=require('/home/zzhang/.local/share/atomx-nist-atlas/browser-check/node_modules/playwright');
const here=__dirname,c=JSON.parse(fs.readFileSync(path.join(here,'contract.json'))),bc=JSON.parse(fs.readFileSync(path.join(here,'browser_contract.json')));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
if(sha(__filename)!==bc.driver_sha256||sha(path.join(here,'contract.json'))!==bc.site_contract_sha256)throw Error('Changed frozen browser preparation');
if(sha(path.join(c.site,c.overlay_path))!==bc.overlay_sha256||sha(path.join(c.site,c.page_path))!==bc.page_sha256)throw Error('Changed site input');
const overlay=JSON.parse(fs.readFileSync(path.join(c.site,c.overlay_path))),old=JSON.parse(fs.readFileSync(path.join(here,'old_models.json')));
const checks=[],errors=[],base=bc.base_url,fmt=n=>n.toLocaleString('en-US',{maximumFractionDigits:8});
const check=(name,passed)=>{checks.push({name,passed:Boolean(passed)});if(!passed)throw Error(name)};
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/home/zzhang/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1080},acceptDownloads:true}),page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  const response=await page.goto(base+'/#Yb:1:models');await page.waitForSelector('.fixed208-summary');
  check('HTTP200_periodic118_models23_rows92',response.status()===200&&await page.locator('.element').count()===118&&await page.locator('.model').count()===23&&await page.locator('.model-lines tbody tr').count()===92);
  check('old_five_runs277points1385states',await page.locator('.correlated-run').count()===5&&await page.locator('.trajectory-points tbody tr').count()===277&&await page.locator('.trajectory-states tbody tr').count()===1385);
  const summary=await page.locator('.fixed208-summary').innerText();
  check('fixed31_1595_originalP_not_optimized',summary.includes('31轨道均未做MCDHF优化')&&summary.includes('1595 CSF')&&summary.includes('其他2194未计算')&&summary.includes('不是原3451度量续算'));
  check('nuclear_counterFAIL_separate389_screened941',summary.includes('14期望/7实际失败')&&summary.includes('389项独立保存数组有限资格通过')&&summary.includes('941项原运行控制通过'));
  check('FD_one_direction_two_h_not_stationarity',summary.includes('7053项运行控制通过')&&summary.includes('一条')&&summary.includes('两个步长')&&summary.includes('不是两条独立方向')&&summary.includes('原29/3451驻值两门仍FAIL'));
  check('five_saved_states_each_visible10',await page.locator('.fixed208-summary table').nth(1).locator('tbody tr').count()===10);
  const fdrows=await page.locator('.fixed208-summary table').nth(3).locator('tbody tr').allTextContents();
  check('FD_literal_originalh_error_gate',overlay.directional_qualification.directions.every((d,i)=>fdrows[i].includes(d.h.toExponential(1))&&fdrows[i].includes(d.error.toExponential(6))&&fdrows[i].includes(d.original_derivative_gate.toExponential(6))));
  for(const model of overlay.models.slice(-2)){
   const block=page.locator('.model[data-model-id="'+model.id+'"]');await block.locator('details > summary').first().click();
   const text=await block.innerText(),rows=await block.locator('.model-lines tbody tr').allTextContents();
   check(model.id+'/fixed_status_scope',text.includes('固定轨道RCI')&&text.includes('未计算扩展驻值')&&text.includes('上下态增量'));
   check(model.id+'/all_four_saved_lines_and_gap',model.rows.every((r,i)=>rows[i].includes(fmt(r.calculated_cm1))&&rows[i].includes(fmt(r.error_cm1))));
   check(model.id+'/original1percentFAIL_unknown_sigma',await block.locator('.model-lines .bad').count()===4&&await block.locator('.model-lines tbody tr').evaluateAll(rs=>rs.every(r=>r.cells[5].textContent==='—')));
   check(model.id+'/no_optimization_trajectory',await page.locator('[data-run-model="'+model.id+'"]').count()===0);
  }
  for(const m of old){const rows=await page.locator('.model[data-model-id="'+(m.id||m.saved_run)+'"] .model-lines tbody tr').allTextContents();check('preserved/'+(m.id||m.saved_run),rows.length===4&&m.rows.every((r,i)=>rows[i].includes(fmt(r.calculated_cm1))))}
  const download=page.waitForEvent('download');await page.locator('a[download][href="updates/yb-correlated29-run003-endpoint16.json"]').click();await(await download).saveAs(path.join(here,'downloaded_update.json'));check('overlay_download_exact',sha(path.join(here,'downloaded_update.json'))===sha(path.join(c.site,c.overlay_path)));
  for(const asset of overlay.source_files.slice(-14)){const r=await context.request.get(base+'/'+asset.asset);check('new_source/'+asset.asset,r.status()===200&&crypto.createHash('sha256').update(await r.body()).digest('hex')===asset.sha256)}
  await page.locator('.fixed208-summary').evaluate(el=>el.scrollIntoView({block:'start'}));await page.screenshot({path:path.join(here,'desktop.png')});await context.close();
  const mobile=await browser.newContext({viewport:{width:390,height:844},isMobile:true,deviceScaleFactor:1}),mp=await mobile.newPage();mp.on('pageerror',e=>errors.push(e.message));await mp.goto(base+'/#Yb:1:models');await mp.waitForSelector('.fixed208-summary');
  check('mobile_models23_rows92_old277points',await mp.locator('.model').count()===23&&await mp.locator('.model-lines tbody tr').count()===92&&await mp.locator('.trajectory-points tbody tr').count()===277);
  check('mobile_fixed_tables_scrollable',await mp.locator('.fixed208-summary .table-wrap').evaluateAll(rs=>rs.every(r=>getComputedStyle(r).overflowX==='auto')));
  await mp.screenshot({path:path.join(here,'mobile.png')});await mobile.close();check('no_JS_errors',errors.length===0);
 }finally{await browser.close()}
 const result={all_passed:true,checks_count:checks.length,checks,javascript_errors:errors,models:23,rows:92,trajectory_rows:277,driver_sha256:sha(__filename),site_contract_sha256:sha(path.join(here,'contract.json')),browser_contract_sha256:sha(path.join(here,'browser_contract.json')),science_NPZ_publication_calls:0};fs.writeFileSync(path.join(here,'browser_result.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({all_passed:true,checks:checks.length,models:23,rows:92,trajectory_rows:277}));
})().catch(e=>{fs.writeFileSync(path.join(here,'browser_failure.json'),JSON.stringify({error:e.stack,checks,errors},null,2)+'\n');console.error(e.stack);process.exitCode=1});
