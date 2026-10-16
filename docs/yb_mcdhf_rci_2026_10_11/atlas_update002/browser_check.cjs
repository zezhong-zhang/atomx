const {chromium}=require('/home/zzhang/.local/share/atomx-nist-atlas/browser-check/node_modules/playwright');
const fs=require('fs');
const path=require('path');
const crypto=require('crypto');
const site='/home/zzhang/.local/share/atomx-nist-atlas/site';
const output=__dirname;
const base=process.argv[2]||'http://127.0.0.1:8893';
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const overlay=JSON.parse(fs.readFileSync(path.join(site,'dist/updates/yb-public531-endpoint160.json'),'utf8'));
const old=JSON.parse(fs.readFileSync(path.join(site,'dist/updates/yb-mcdhf-stage002.json'),'utf8'));
const checks=[];
const errors=[];
const check=(name,passed)=>{checks.push({name,passed:!!passed});if(!passed)throw Error(name)};
let browser;

(async()=>{
  browser=await chromium.launch({headless:true,executablePath:'/home/zzhang/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'});
  const context=await browser.newContext({viewport:{width:1440,height:1080},acceptDownloads:true});
  const page=await context.newPage();
  page.on('pageerror',error=>errors.push(error.message));
  const response=await page.goto(base+'/#Yb:1:models');
  await page.waitForSelector('.endpoint-summary');
  check('HTTP200',response.status()===200);
  check('118_elements',await page.locator('.element').count()===118);
  check('12_models_48_rows',await page.locator('.model').count()===12&&await page.locator('.model-lines tbody tr').count()===48);
  check('2_new_endpoint_models',await page.locator('.model[data-model-kind="nonstationary_endpoint_rci"]').count()===2);
  const current=page.locator('.endpoint-summary');
  check('actual_original_residual_fail_and_energy_pass_visible',await current.textContent().then(t=>t.includes('4.717048e-5')&&t.includes('1.0e-7')&&t.includes('6.889724e-9')&&t.includes('1.0e-8')&&t.includes('不能替代驻值门')));
  check('complete_physics_and_proof_boundary_visible',await current.textContent().then(t=>t.includes('531')&&t.includes('69')&&t.includes('24')&&t.includes('16')&&t.includes('3,710')&&t.includes('不是已收敛MCDHF')&&t.includes('未加动态Breit')));
  check('absolute_common_shift_and_experimental_gap_visible',await current.textContent().then(t=>t.includes('14.14834612')&&t.includes('1708–2313')&&t.includes('该绝对修正不是谱线改变量')));
  for(const model of overlay.models){
    const locator=page.locator('[data-model-id="'+model.id+'"]');
    if(!await locator.locator('details').evaluate(el=>el.open))await locator.locator('summary').click();
    check(model.id+'/nonstationary_badge',await locator.locator('.badge').textContent().then(t=>t.includes('原DC驻值未过')));
    check(model.id+'/four_nominal_goals_fail',await locator.locator('.model-lines .bad').count()===4);
    const displayed=await locator.locator('.model-lines tbody tr').evaluateAll(rows=>rows.map(row=>Array.from(row.querySelectorAll('td')).map(cell=>cell.textContent)));
    check(model.id+'/four_actual_labeled_rows',displayed.length===4&&displayed.every((cells,i)=>cells[0].includes(model.rows[i].configuration)&&cells[0].includes(String(model.rows[i].two_j))&&Math.abs(Number(cells[1].replaceAll(',',''))-model.rows[i].calculated_cm1)<1e-7&&Math.abs(Number(cells[2].replaceAll(',',''))-model.rows[i].reference_cm1)<1e-7));
    check(model.id+'/physical_overlap_not_bare_weight',await locator.textContent().then(t=>t.includes('原physical参考重叠²')&&!t.includes('bare连接权重')));
  }
  const latest=page.locator('[data-model-id="public531_endpoint160_static_Breit"]');
  check('latest_samepoint_upper_lower_and_shift_visible',await latest.textContent().then(t=>t.includes('同一accepted160点')&&t.includes('下态 ΔHa')&&t.includes('谱线 Δcm')));
  await current.locator('details summary').click();
  check('centroid_and_splitting_actual_numbers_visible',await current.textContent().then(t=>['25,733.79393973','766.42624129','27,273.93372302','3,475.15502322','-31.13542224','-23.56473697'].every(x=>t.includes(x))));
  for(const model of old.models){
    const locator=page.locator('[data-model-id="'+model.id+'"]');
    check('prior_model_exact_energies/'+model.id,await locator.locator('.model-lines tbody tr').evaluateAll((rows,expected)=>rows.length===4&&rows.every((row,i)=>Math.abs(Number(row.querySelectorAll('td')[1].textContent.replaceAll(',',''))-expected[i])<1e-7),model.rows.map(row=>row.calculated_cm1)));
  }
  check('prior_failed_one_step_probe_retained',await page.locator('[data-probe-status]').textContent().then(t=>t.includes('0.03676072')&&t.includes('0.0249933')&&t.includes('不修改')));
  check('historical_SD_and_dynamic_models_retained',await page.locator('[data-model-id="yb_full_monovalent006"]').count()===1&&await page.locator('[data-model-id="yb_dynamic_breit003"]').count()===1);
  await current.evaluate(el=>el.scrollIntoView({block:'start'}));
  await page.screenshot({path:path.join(output,'desktop.png'),fullPage:false});
  const downloadPromise=page.waitForEvent('download');
  await page.locator('a[href="updates/yb-public531-endpoint160.json"][download]').click();
  const download=await downloadPromise;
  await download.saveAs(path.join(output,'downloaded_update.json'));
  check('new_overlay_download_exact_SHA',sha(path.join(output,'downloaded_update.json'))===sha(path.join(site,'dist/updates/yb-public531-endpoint160.json')));
  await page.locator('[data-tab="sources"]').click();
  check('new_old_original_source_chains_visible',await page.locator('#content').textContent().then(t=>t.includes('公共531 accepted160')&&t.includes('saved_array_verification.json')&&t.includes('independent_endpoint_verification.json')&&t.includes('balanced_cv_replay/verification.json')&&t.includes('NIST')));
  for(const source of overlay.source_files){
    const assetResponse=await page.request.get(base+'/'+source.asset);
    check('exact_science_asset_access/'+source.asset,assetResponse.status()===200&&crypto.createHash('sha256').update(await assetResponse.body()).digest('hex')===source.sha256);
  }
  await page.locator('[data-tab="levels"]').click();
  await page.waitForFunction(()=>document.querySelector('#page-label').textContent.includes('349'));
  check('original349_level_pagination',await page.locator('#content tbody tr').count()===75);
  await page.locator('#next').click();
  check('original_next_page',await page.locator('#page-label').textContent().then(t=>t.includes('2 /')));
  await page.locator('[data-z="10"]').click();
  await page.waitForFunction(()=>document.querySelector('#ion-info').textContent.startsWith('Ne I'));
  await page.locator('[data-tab="lines"]').click();
  await page.waitForFunction(()=>document.querySelector('#content').textContent.includes('实测 nm'));
  await page.locator('#kind').selectOption('observed');
  check('Ne_observed_filter_retained',await page.locator('#content tbody tr').count()>0);
  await page.locator('[data-z="70"]').click();
  await page.locator('#ion').selectOption('1');
  await page.locator('[data-tab="models"]').click();
  await page.waitForSelector('.endpoint-summary');
  check('cached_revisit_no_duplicate',await page.locator('.model').count()===12&&await page.locator('.model-lines tbody tr').count()===48);
  await page.locator('#search').fill('Yb');
  check('periodic_search_retained',await page.locator('.element:not(.dim)').count()===1);
  await page.setViewportSize({width:390,height:844});
  await page.locator('.endpoint-summary').evaluate(el=>el.scrollIntoView({block:'start'}));
  check('mobile_endpoint_status_visible',await page.locator('.endpoint-summary').textContent().then(t=>t.includes('原轨道驻值仍未通过')&&t.includes('accepted160')&&t.includes('不是已收敛MCDHF')));
  check('mobile_page_fits_viewport',await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1));
  await page.screenshot({path:path.join(output,'mobile.png'),fullPage:false});
  await page.locator('#search').fill('');
  await page.locator('[data-z="118"]').scrollIntoViewIfNeeded();
  await page.locator('[data-z="118"]').click();
  await page.waitForFunction(()=>document.querySelector('#title').textContent.startsWith('Og'));
  await page.locator('[data-tab="models"]').click();
  await page.waitForFunction(()=>document.querySelector('#content').textContent.includes('没有接入'));
  check('missing_species_retained',await page.locator('.endpoint-summary').count()===0);
  const fallbackContext=await browser.newContext();
  const fallback=await fallbackContext.newPage();
  fallback.on('pageerror',error=>errors.push(error.message));
  await fallback.route('**/updates/yb-public531-endpoint160.json',route=>route.abort());
  await fallback.goto(base+'/#Yb:1:models');
  await fallback.waitForFunction(()=>document.querySelector('#content').textContent.includes('公共531端点对照读取失败'));
  check('new_overlay_failure_keeps_prior10models40rows',await fallback.locator('.model').count()===10&&await fallback.locator('.model-lines tbody tr').count()===40&&await fallback.locator('.endpoint-summary').count()===0);
  const historicalContext=await browser.newContext();
  const historical=await historicalContext.newPage();
  historical.on('pageerror',error=>errors.push(error.message));
  await historical.route('**/updates/*.json',route=>route.abort());
  await historical.goto(base+'/#Yb:1:models');
  await historical.waitForFunction(()=>document.querySelector('#content').textContent.includes('公共531端点对照读取失败'));
  check('both_overlay_failure_keeps_historical2models8rows',await historical.locator('.model').count()===2&&await historical.locator('.model-lines tbody tr').count()===8);
  check('no_JS_errors',errors.length===0);
  const report={all_passed:checks.every(check=>check.passed),checks_count:checks.length,checks,javascript_errors:errors,
    page_sha256:sha(path.join(site,'dist/index.html')),overlay_sha256:sha(path.join(site,'dist/updates/yb-public531-endpoint160.json')),
    source_sha256:sha(__filename),browser:'Existing cached Playwright Chromium, owned loopback local-only static server',
    new_matched_rows:8,prior_matched_rows:40,screenshots:{desktop:sha(path.join(output,'desktop.png')),mobile:sha(path.join(output,'mobile.png'))},
    publication_performed:false,science_calls:0};
  fs.writeFileSync(path.join(output,'browser_result.json'),JSON.stringify(report,null,2)+'\n');
  await browser.close();
  process.stdout.write(JSON.stringify({all_passed:report.all_passed,checks_count:checks.length,javascript_errors:errors})+'\n');
})().catch(async error=>{
  fs.writeFileSync(path.join(output,'browser_failure.json'),JSON.stringify({message:error.message,stack:error.stack,completed_checks:checks,javascript_errors:errors,source_sha256:sha(__filename)},null,2)+'\n');
  if(browser)await browser.close();
  process.stderr.write(error.stack+'\n');
  process.exit(1);
});
