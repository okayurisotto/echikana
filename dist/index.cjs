"use strict";var T=Object.create;var u=Object.defineProperty;var k=Object.getOwnPropertyDescriptor;var B=Object.getOwnPropertyNames;var C=Object.getPrototypeOf,I=Object.prototype.hasOwnProperty;var L=(e,s)=>{for(var n in s)u(e,n,{get:s[n],enumerable:!0})},y=(e,s,n,r)=>{if(s&&typeof s=="object"||typeof s=="function")for(let t of B(s))!I.call(e,t)&&t!==n&&u(e,t,{get:()=>s[t],enumerable:!(r=k(s,t))||r.enumerable});return e};var z=(e,s,n)=>(n=e!=null?T(C(e)):{},y(s||!e||!e.__esModule?u(n,"default",{value:e,enumerable:!0}):n,e)),P=e=>y(u({},"__esModule",{value:!0}),e);var E={};L(E,{EchikanaInferencer:()=>l,errors:()=>S});module.exports=P(E);var c=z(require("onnxruntime-node"),1),A=z(require("sharp"),1);var v=e=>{let s=e.map(n=>Math.exp(n)).reduce((n,r)=>n+r,0);return e.map(n=>Math.exp(n)/s)};var a=class extends Error{constructor(){super("The inference model has not been initialized.")}},i=class extends Error{constructor(){super("An internal error has occurred.")}},l=class{constructor(s){this.model=s}size=224;output="logits";session=null;get initialized(){return this.session!==null}async initialize(){this.initialized&&await this.dispose(),this.session=await c.default.InferenceSession.create(this.model)}async inference(s){if(this.initialized||await this.initialize(),this.session===null)throw new a;let n=await(0,A.default)(s).resize(this.size,this.size,{kernel:"nearest",fit:"fill"}).removeAlpha().raw().toBuffer(),r=[],t=[],d=[];for(let o=0;o<n.byteLength;o+=3){let w=n[o+0];if(w===void 0)throw new i;r.push(w);let b=n[o+1];if(b===void 0)throw new i;t.push(b);let x=n[o+2];if(x===void 0)throw new i;d.push(x)}let g=[...r,...t,...d].map(o=>(o/255-.5)/.5),f=new c.default.Tensor(new Float32Array(g),[1,3,this.size,this.size]),p=(await this.session.run({pixel_values:f}))[this.output];if(p===void 0)throw new i;let h=p.data;if(!(h instanceof Float32Array))throw new i;let[,m]=v([...h]);if(m===void 0)throw new i;return f.dispose(),p.dispose(),m}async dispose(){await this.session?.release()}};var S={InitializationError:a,InternalError:i};0&&(module.exports={EchikanaInferencer,errors});