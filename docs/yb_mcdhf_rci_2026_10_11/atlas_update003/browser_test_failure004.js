'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {chromium}=require('/home/zzhang/.local/share/atomx-nist-atlas/browser-check/node_modules/playwright');
const here=__dirname;
const contract=JSON.parse(fs.readFileSync(path.join(here,'contract.json')));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const load=p=>JSON.parse(fs.readFileSync(p));
const overlay=load(path.join(contract.site,contract.new_overlay));
const old=[...load(path.join(contract.site,'dist/updates/yb-public531-endpoint160.json')).models,...load(path.join(contract.site,'dist/updates/yb-mcdhf-stage002.json')).models];
const checks=[],errors=[];
const check=(name,passed)=>checks.push({name,passed:Boolean(passed)});
const base='http://127.0.0.1:39317';
const fmt=n=>n.toLocaleString('en-US',{maximumFractionDigits:8});
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/home/zzhang/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1080},acceptDownloads:true});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  const response=await page.goto(base+'/#Yb:1:models');
  await page.waitForSelector('.iteration-summary');
  check('HTTP200',response.status()===200);
  check('118periodic_elements',await page.locator('.element').count()===118);
  check('14models_56lines',await page.locator('.model').count()===14 && await page.locator('.model-lines tbody tr').count()===56);
  const summary=await page.locator('.iteration-summary').innerText();
  for(const [name,text] of [['stationarityFAIL','1.943485e-6'],['full531','完整531 CSF'],['five_complete_targetsectors','五个完整目标J/宇称扇区1240'],['distinctpoints','两个新模型分别属于320点和160点'],['negative_rank','增至'],['RMSbaseline','2,022.82969562'],['RMSrank','2,225.97408'],['no_splice','没有与旧静态Breit']])check('summary/'+name,summary.includes(text));
  for(const m of overlay.models){
   const selector='.model[data-model-id="'+m.id+'"]';
   const block=page.locator(selector);const text=await block.innerText();
   check(m.id+'/correct_status',text.includes(m.id==='fixed160_5p_double_DC'?'扩展驻值未计算':'原驻值未过'));
   check(m.id+'/four_failed_nominal_goals',await block.locator('.model-lines .bad').count()===4);
   check(m.id+'/physical_overlap_not_bare',text.includes('原physical参考重叠²')&&!text.includes('bare连接权重'));
   const rows=await block.locator('.model-lines tbody tr').allTextContents();
   check(m.id+'/explicit_four_label_order',rows.length===4 && m.rows.every((r,i)=>rows[i].includes(r.configuration+' / '+r.two_j)));
   check(m.id+'/actual_energy_and_NIST_rows',m.rows.every((r,i)=>rows[i].includes(fmt(r.calculated_cm1))&&rows[i].includes(fmt(r.reference_cm1))&&rows[i].includes(fmt(r.error_cm1))));
   check(m.id+'/upper_lower_difference_visible',text.includes('上态 ΔHa')&&text.includes('下态 ΔHa')&&text.includes('谱线 Δcm⁻¹'));
   check(m.id+'/centroid_split_visible',text.includes('重心')&&text.includes('分裂'));
   check(m.id+'/SHA_visible',text.includes(m.sha256));
  }
  for(const m of old){
   const selector='.model[data-model-id="'+(m.id||m.saved_run)+'"]';
   const texts=await page.locator(selector+' .model-lines tbody tr').allTextContents();
   check('prior_exact_values/'+(m.id||m.saved_run),texts.length===4&&m.rows.every((r,i)=>texts[i].includes(fmt(r.calculated_cm1))));
  }
  check('historical2models_retained',await page.locator('.model[data-model-kind="historical"]').count()===2);
  check('old_static160_scope_heading',await page.locator('.endpoint-summary').innerText().then(t=>t.includes('此前160点')&&t.includes('不是已收敛MCDHF')));
  check('old_failed_probe_retained',await page.locator('.current-summary:not(.endpoint-summary):not(.iteration-summary)').innerText().then(t=>t.includes('驻值')&&t.includes('一步')));
  const downloadPromise=page.waitForEvent('download');await page.locator('a[download][href="updates/yb-public531-endpoint320.json"]').click();
  const download=await downloadPromise;await download.saveAs(path.join(here,'downloaded_update.json'));
  check('new_overlay_download_exactSHA',sha(path.join(here,'downloaded_update.json'))===sha(path.join(contract.site,contract.new_overlay)));
  await page.locator('[data-tab="sources"]').click();
  check('both_new_and_old_source_chains',await page.locator('#content').innerText().then(t=>t.includes('320点DC')&&t.includes('accepted160')&&t.includes('近期 MCDHF')));
  for(const item of overlay.source_files){
   const res=await context.request.get(base+'/'+item.asset);
   const body=await res.body();check('downloaded_exact_source/'+item.asset,res.status()===200&&crypto.createHash('sha256').update(body).digest('hex')===item.sha256);
  }
  await page.locator('[data-tab="levels"]').click();
  check('349Yb_reference_levels',await page.locator('#page-label').innerText().then(t=>t.includes('349 条')));
  const first=await page.locator('#content tbody tr').first().innerText();await page.locator('#next').click();
  check('Yb_pagination',await page.locator('#page-label').innerText().then(t=>t.includes('2 /'))&&await page.locator('#content tbody tr').first().innerText()!==first);
  await page.locator('[data-z="10"]').click();await page.waitForFunction(()=>location.hash.startsWith('#Ne:0:'));
  await page.locator('[data-tab="lines"]').click();await page.locator('#kind').selectOption('observed');
  check('Ne_observed_filter',await page.locator('#content tbody tr').count()>0&&await page.locator('#content tbody tr').evaluateAll(rs=>rs.every(r=>r.cells[2].textContent!=='—')));
  await page.locator('#kind').selectOption('all');await page.locator('[data-z="70"]').click();
  await page.locator('#ion').selectOption('1');await page.locator('[data-tab="models"]').click();await page.waitForSelector('.iteration-summary');
  check('cached_revisit_no_duplicates',await page.locator('.model').count()===14&&await page.locator('.model-lines tbody tr').count()===56);
  await page.locator('#search').fill('Yb');check('periodic_search',await page.locator('.element:not(.dim)').count()===1);
  await page.locator('#search').fill('');await page.locator('[data-z="118"]').click();await page.waitForFunction(()=>location.hash.startsWith('#Og:0:'));
  await page.locator('[data-tab="models"]').click();check('missing_reference_explicit',await page.locator('#content').innerText().then(t=>t.includes('没有')||t.includes('无')));
  await page.goto(base+'/#Yb:1:models');await page.waitForSelector('.iteration-summary');await page.locator('.iteration-summary').scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(here,'desktop.png')});
  const mobile=await browser.newContext({viewport:{width:390,height:844},isMobile:true,deviceScaleFactor:1});
  const mp=await mobile.newPage();mp.on('pageerror',e=>errors.push(e.message));await mp.goto(base+'/#Yb:1:models');await mp.waitForSelector('.iteration-summary');
  check('mobile14models56rows',await mp.locator('.model').count()===14&&await mp.locator('.model-lines tbody tr').count()===56);
  check('mobile_table_overflow_accessible',await mp.locator('.iteration-summary .table-wrap').evaluateAll(rs=>rs.every(r=>getComputedStyle(r).overflowX==='auto')));
  await mp.locator('.iteration-summary').scrollIntoViewIfNeeded();await mp.screenshot({path:path.join(here,'mobile.png')});await mobile.close();
  const fallback=await browser.newContext();const fp=await fallback.newPage();fp.on('pageerror',e=>errors.push(e.message));
  await fp.route('**/updates/yb-public531-endpoint320.json',route=>route.fulfill({status:503,body:'temporarily unavailable'}));
  await fp.goto(base+'/#Yb:1:models');await fp.waitForSelector('.endpoint-summary');
  check('new_overlay_failure_retains12models48rows',await fp.locator('.model').count()===12&&await fp.locator('.model-lines tbody tr').count()===48);
  check('new_overlay_failure_visible_no_new_cert',await fp.locator('#content').innerText().then(t=>t.includes('最新两项对照读取失败'))&&await fp.locator('.iteration-summary').count()===0);
  await fallback.close();
  const historical=await browser.newContext();const hp=await historical.newPage();hp.on('pageerror',e=>errors.push(e.message));
  await hp.route('**/updates/*.json',route=>route.fulfill({status:503,body:'unavailable'}));
  await hp.goto(base+'/#Yb:1:models');await hp.waitForSelector('.model');
  check('all_overlay_failure_retains2historical8rows',await hp.locator('.model').count()===2&&await hp.locator('.model-lines tbody tr').count()===8);
  await historical.close();
  check('no_javascript_errors',errors.length===0);await context.close();
 }finally{await browser.close()}
 const report={all_passed:checks.every(x=>x.passed),checks_count:checks.length,checks,javascript_errors:errors,page_sha256:sha(path.join(contract.site,'dist/index.html')),overlay_sha256:sha(path.join(contract.site,contract.new_overlay)),source_sha256:sha(__filename),contract_sha256:sha(path.join(here,'contract.json')),browser:'Existing cached Playwright Chromium1243; desktop1440x1080/mobile390x844',models:14,rows:56,old_rows_preserved:48,screenshots:['desktop.png','mobile.png'],publication_performed:false,science_calls:0};
 fs.writeFileSync(path.join(here,'browser_result.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify({all_passed:report.all_passed,checks_count:checks.length,failed:checks.filter(x=>!x.passed),errors}));
 if(!report.all_passed)process.exitCode=1;
})().catch(e=>{fs.writeFileSync(path.join(here,'browser_failure.json'),JSON.stringify({error:e.stack,checks,errors},null,2)+'\n');console.error(e.stack);process.exitCode=1});
