import x from"onnxruntime-node";import z from"sharp";var b=i=>{let r=i.map(e=>Math.exp(e)).reduce((e,o)=>e+o,0);return i.map(e=>Math.exp(e)/r)};var t=class extends Error{constructor(){super("The inference model has not been initialized.")}},n=class extends Error{constructor(){super("An internal error has occurred.")}},u=class{constructor(r){this.model=r}size=224;output="logits";session=null;get initialized(){return this.session!==null}async initialize(){this.initialized&&await this.dispose(),this.session=await x.InferenceSession.create(this.model)}async inference(r){if(this.initialized||await this.initialize(),this.session===null)throw new t;let e=await z(r).resize(this.size,this.size,{kernel:"nearest",fit:"fill"}).removeAlpha().raw().toBuffer(),o=[],l=[],p=[];for(let s=0;s<e.byteLength;s+=3){let h=e[s+0];if(h===void 0)throw new n;o.push(h);let m=e[s+1];if(m===void 0)throw new n;l.push(m);let w=e[s+2];if(w===void 0)throw new n;p.push(w)}let y=[...o,...l,...p].map(s=>(s/255-.5)/.5),c=new x.Tensor(new Float32Array(y),[1,3,this.size,this.size]),a=(await this.session.run({pixel_values:c}))[this.output];if(a===void 0)throw new n;let d=a.data;if(!(d instanceof Float32Array))throw new n;let[,f]=b([...d]);if(f===void 0)throw new n;return c.dispose(),a.dispose(),f}async dispose(){await this.session?.release()}};var L={InitializationError:t,InternalError:n};export{u as EchikanaInferencer,L as errors};
