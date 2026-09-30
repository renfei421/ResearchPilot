// Offline DOM harness: execute the shipped handlers, not a copy of their logic.
const fs = require('fs'), vm = require('vm'), path = require('path'), assert = require('assert');
class Element {
  constructor(id='') {this.id=id; this.value=''; this.checked=false; this.hidden=false; this.children=[]; this.listeners={}; this.dataset={};}
  addEventListener(name, fn) {(this.listeners[name] ||= []).push(fn);}
  async emit(name, overrides={}) {for(const fn of this.listeners[name] || []) await fn({preventDefault(){}, submitter:new Element(), ...overrides});}
  dispatchEvent(e) {return this.emit(e.type);}
  append(...items) {this.children.push(...items);}
  replaceChildren(...items) {this.children=items;}
  focus() {} scrollIntoView() {} reset() {}
}
const elements=new Map(); const el=id=>{if(!elements.has(id)) elements.set(id,new Element(id)); return elements.get(id);};
const calls=[], projects=[]; let runCounter=0;
function response(data) {return {ok:true,json:async()=>data,text:async()=>typeof data==='string'?data:JSON.stringify(data)};}
async function fetch(url, options={}) {
  const body=options.body?JSON.parse(options.body):null; calls.push({url, ...options, body});
  if(url==='/health') return response({openai_configured:true});
  if(url.startsWith('/research?')) return response([]);
  if(url==='/projects' && options.method==='POST') {const p={...body,project_id:'p1',status:'active'}; projects.push(p); return response(p);}
  if(url==='/projects') return response(projects);
  if(url==='/projects/p1/view') return response('<h2>Safe server HTML</h2>');
  if(url==='/projects/p1' && options.method==='PATCH') {Object.assign(projects[0],body); return response(projects[0]);}
  if(url==='/projects/p1') return response({project:projects[0]});
  if((url==='/research' || url==='/claim-check' || url==='/projects/p1/continue') && options.method==='POST') return response({run_id:'r'+(++runCounter)});
  if(/^\/research\/r/.test(url)) return response({question:'Q',run_id:url.split('/').pop(),status:'running',current_stage:'searching',current_round:1,selected_papers:0,evidence_collected:0,warnings:[],updated_at:new Date().toISOString()});
  throw Error('Unexpected offline request '+url);
}
const sandbox={document:{getElementById:el,createElement:()=>new Element()}, window:{}, fetch, console,
  Option:class extends Element{constructor(text,value){super();this.textContent=text;this.value=value;}},
  Event:class{constructor(type){this.type=type;}}, URLSearchParams, history:{replaceState(){}},
  location:{pathname:'/',search:''}, setTimeout:()=>1, clearTimeout(){}};
vm.createContext(sandbox);
const root=path.join(__dirname,'../src/researchpilot/web');
vm.runInContext(fs.readFileSync(path.join(root,'app.js'),'utf8'),sandbox);
vm.runInContext(fs.readFileSync(path.join(root,'projects.js'),'utf8'),sandbox);
const flush=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
  await flush();
  el('project-title').value='Workspace';el('project-description').value='Scope';el('project-field').value='Math';
  await el('project-create').emit('submit');
  assert.equal(projects[0].title,'Workspace'); assert.equal(el('run-project').value,'p1');
  assert.equal(el('project-panel').hidden,false);
  await el('project-ask').emit('click'); el('question').value='Research question';
  await el('research-form').emit('submit'); await flush();
  const research=calls.find(x=>x.url==='/research'&&x.method==='POST');
  assert.equal(research.body.project_id,'p1');assert.equal(research.body.question,'Research question');
  await el('project-idea').emit('click');el('question').value='Idea claim';
  await el('research-form').emit('submit');await flush();
  const idea=calls.find(x=>x.url==='/claim-check'&&x.method==='POST');
  assert.equal(idea.body.project_id,'p1');assert.equal(idea.body.claim,'Idea claim');
  const button=new Element();button.dataset={continue:'gap',id:'g1',mode:'research'};
  button.closest=()=>({querySelector:()=>({textContent:'Remaining gap'})});
  await el('project-detail').emit('click',{target:{closest:s=>s==='button'?button:null}});
  el('project-continue-instruction').value='Investigate';
  await el('project-continue-form').emit('submit');await flush();
  const continuation=calls.find(x=>x.url==='/projects/p1/continue'&&x.method==='POST');
  assert.equal(continuation.body.target_id,'g1');assert.equal(continuation.body.instruction,'Investigate');
  await el('project-archive').emit('click');assert.equal(projects[0].status,'archived');assert.equal(el('project-ask').disabled,true);
  console.log('create, research, idea, continue, archive: passed');
})().catch(err=>{console.error(err);process.exitCode=1;});
