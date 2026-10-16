'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {chromium}=require('/home/zzhang/.local/share/atomx-nist-atlas/browser-check/node_modules/playwright');
const here=__dirname,contract=JSON.parse(fs.readFileSync(path.join(here,'contract.json')));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const overlay=JSON.parse(fs.readFileSync(path.join(contract.site,contract.new_overlay))),old=JSON.parse(fs.readFileSync(path.join(here,'old_models.json')));
const checks=[],errors=[],base='http://127.0.0.1:39318',fmt=n=>n.toLocaleString('en-US',{maximumFractionDigits:8});
const check=(name,passed)=>{checks.push({name,passed:Boolean(passed)});if(!passed)throw Error(name)};
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/home/zzhang/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1080},acceptDownloads:true}),page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  const response=await page.goto(base+'/#Yb:1:models');await page.waitForSelector('.correlated-summary');
  check('HTTP200_periodic118_19models76rows',response.status()===200&&await page.locator('.element').count()===118&&await page.locator('.model').count()===19&&await page.locator('.model-lines tbody tr').count()===76);
  check('complete_three_runs211points1055states',await page.locator('.trajectory-points tbody tr').count()===211&&await page.locator('.trajectory-states tbody tr').count()===1055);
  const model=overlay.models[0],block=page.locator('.model[data-model-id="'+model.id+'"]'),article=page.locator('[data-run-model="'+model.id+'"]');
  const summary=await article.innerText();check('128new208global_uncontrolled_originalFAIL',summary.includes('本轮 128 个更新')&&summary.includes('累计 208 个更新')&&summary.includes('129 个接受点')&&summary.includes('不是已驻值结果')&&summary.includes('未受控墙钟时间'));
  check('all_four_original_experimentFAIL',await block.locator('.model-lines .bad').count()===4);
  const rows=await block.locator('.model-lines tbody tr').allTextContents();check('terminal_four_lines_gap_values',model.rows.every((r,i)=>rows[i].includes(fmt(r.calculated_cm1))&&rows[i].includes(fmt(r.error_cm1))));
  check('missing_sigma_remains_empty',await block.locator('.model-lines tbody tr').evaluateAll(rs=>rs.every(r=>r.cells[5].textContent==='—')));
  await article.locator('.trajectory-details > summary').click();check('all129points_visible',await article.locator('.trajectory-points').isVisible()&&await article.locator('.trajectory-points tbody tr').count()===129);
  const last=await article.locator('.trajectory-points tbody tr').last().innerText();check('predeclared_terminal_RMS_EOL_residual',last.includes(fmt(model.nominal_gap_RMS_cm1))&&last.includes(fmt(model.EOL_Ha))&&last.includes(model.orbital_residual.toExponential(8)));
  for(const m of old){const values=await page.locator('.model[data-model-id="'+(m.id||m.saved_run)+'"] .model-lines tbody tr').allTextContents();check('old_values/'+(m.id||m.saved_run),values.length===4&&m.rows.every((r,i)=>values[i].includes(fmt(r.calculated_cm1))))}
  const downloadEvent=page.waitForEvent('download');await page.locator('a[download][href="updates/yb-correlated29-run003-endpoint16.json"]').click();await(await downloadEvent).saveAs(path.join(here,'downloaded_update.json'));check('overlay_download_exact',sha(path.join(here,'downloaded_update.json'))===sha(path.join(contract.site,contract.new_overlay)));
  for(const asset of overlay.source_files.slice(0,6)){const response=await context.request.get(base+'/'+asset.asset);check('new_source/'+asset.asset,response.status()===200&&crypto.createHash('sha256').update(await response.body()).digest('hex')===asset.sha256)}
  await article.evaluate(el=>el.scrollIntoView({block:'start'}));await page.screenshot({path:path.join(here,'desktop.png')});await context.close();
  const mobile=await browser.newContext({viewport:{width:390,height:844},isMobile:true,deviceScaleFactor:1}),mp=await mobile.newPage();mp.on('pageerror',e=>errors.push(e.message));await mp.goto(base+'/#Yb:1:models');await mp.waitForSelector('.correlated-summary');
  check('mobile19models76rows',await mp.locator('.model').count()===19&&await mp.locator('.model-lines tbody tr').count()===76);check('mobile_trajectory_tables_scrollable',await mp.locator('.correlated-summary .table-wrap').evaluateAll(rs=>rs.every(r=>getComputedStyle(r).overflowX==='auto')));await mp.locator('[data-run-model="'+model.id+'"]').evaluate(el=>el.scrollIntoView({block:'start'}));await mp.screenshot({path:path.join(here,'mobile.png')});await mobile.close();check('no_JS_errors',errors.length===0);
 }finally{await browser.close()}
 const report={all_passed:true,checks_count:checks.length,checks,javascript_errors:errors,models:19,rows:76,trajectory_rows:211,page_sha256:sha(path.join(contract.site,'dist/index.html')),overlay_sha256:sha(path.join(contract.site,contract.new_overlay)),driver_sha256:sha(__filename),contract_sha256:sha(path.join(here,'contract.json')),science_calls:0,publication_performed:false};fs.writeFileSync(path.join(here,'browser_result.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({all_passed:true,checks:checks.length,models:19,rows:76,trajectory_rows:211}));
})().catch(e=>{fs.writeFileSync(path.join(here,'browser_failure.json'),JSON.stringify({error:e.stack,checks,errors},null,2)+'\n');console.error(e.stack);process.exitCode=1});
