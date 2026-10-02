import {readFileSync,existsSync,readdirSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import assert from 'node:assert/strict';
const html=readFileSync('frontend/index.html','utf8'),ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
assert.equal(ids.length,new Set(ids).size,'Duplicate HTML IDs');
for(const name of readdirSync('frontend').filter(f=>f.endsWith('.js'))){execFileSync(process.execPath,['--check','frontend/'+name]);if(name==='app.js'||name==='workbench.js'){const source=readFileSync('frontend/'+name,'utf8');for(const match of source.matchAll(/\$\('([^']+)'\)/g))assert.ok(ids.includes(match[1]),'Missing element '+match[1]);}}
for(const match of html.matchAll(/(?:src|href)="([^"#]+)"/g)){const path=match[1];if(!/^(https?:|data:|mailto:)/.test(path))assert.ok(existsSync('frontend/'+path),'Missing asset '+path);}
for(const section of ['workspace','batch','map','ecology','samples','projects','method'])assert.ok(ids.includes(section));

console.log('JavaScript syntax, DOM elements, bundled assets and Pages configuration verified.');
