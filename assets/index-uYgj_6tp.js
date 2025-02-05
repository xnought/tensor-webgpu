(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))n(s);new MutationObserver(s=>{for(const i of s)if(i.type==="childList")for(const a of i.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&n(a)}).observe(document,{childList:!0,subtree:!0});function r(s){const i={};return s.integrity&&(i.integrity=s.integrity),s.referrerPolicy&&(i.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?i.credentials="include":s.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function n(s){if(s.ep)return;s.ep=!0;const i=r(s);fetch(s.href,i)}})();const ot=!1;var Et=Array.isArray,tr=Array.prototype.indexOf,Ne=Array.from,rr=Object.defineProperty,lt=Object.getOwnPropertyDescriptor,nr=Object.getOwnPropertyDescriptors,sr=Object.getPrototypeOf;function ir(t){return t()}function ft(t){for(var e=0;e<t.length;e++)t[e]()}const G=2,At=4,Be=8,Le=16,H=32,fe=64,me=128,N=256,ye=512,T=1024,z=2048,ie=4096,W=8192,Pe=16384,ar=32768,We=65536,ur=1<<19,$t=1<<20,ct=Symbol("$state");function It(t){return t===this.v}function or(t,e){return t!=t?e==e:t!==e||t!==null&&typeof t=="object"||typeof t=="function"}function Bt(t){return!or(t,this.v)}function lr(t){throw new Error("https://svelte.dev/e/effect_in_teardown")}function fr(){throw new Error("https://svelte.dev/e/effect_in_unowned_derived")}function cr(t){throw new Error("https://svelte.dev/e/effect_orphan")}function dr(){throw new Error("https://svelte.dev/e/effect_update_depth_exceeded")}function pr(){throw new Error("https://svelte.dev/e/state_unsafe_local_read")}function hr(){throw new Error("https://svelte.dev/e/state_unsafe_mutation")}let ce=!1,_r=!1;function gr(){ce=!0}const vr=1,mr=2,yr=16,xr=1,wr=2,br=Symbol();function Or(t){throw new Error("https://svelte.dev/e/lifecycle_outside_component")}let b=null;function dt(t){b=t}function Pt(t,e=!1,r){b={p:b,c:null,e:null,m:!1,s:t,x:null,l:null},ce&&!e&&(b.l={s:null,u:null,r1:[],r2:xe(!1)})}function St(t){const e=b;if(e!==null){const a=e.e;if(a!==null){var r=w,n=x;e.e=null;try{for(var s=0;s<a.length;s++){var i=a[s];se(i.effect),ne(i.reaction),Lt(i.fn)}}finally{se(r),ne(n)}}b=e.p,e.m=!0}return{}}function Se(){return!ce||b!==null&&b.l===null}function xe(t,e){var r={f:0,v:t,reactions:null,equals:It,rv:0,wv:0};return r}function Dt(t,e=!1){var n;const r=xe(t);return e||(r.equals=Bt),ce&&b!==null&&b.l!==null&&((n=b.l).s??(n.s=[])).push(r),r}function pt(t,e=!1){return Er(Dt(t,e))}function Er(t){return x!==null&&!q&&x.f&G&&(C===null?Cr([t]):C.push(t)),t}function Me(t,e){return x!==null&&!q&&Se()&&x.f&(G|Le)&&(C===null||!C.includes(t))&&hr(),Tt(t,e)}function Tt(t,e){return t.equals(e)||(t.v,t.v=e,t.wv=Kt(),kt(t,z),Se()&&w!==null&&w.f&T&&!(w.f&(H|fe))&&(L===null?Nr([t]):L.push(t))),e}function kt(t,e){var r=t.reactions;if(r!==null)for(var n=Se(),s=r.length,i=0;i<s;i++){var a=r[i],u=a.f;u&z||!n&&a===w||(j(a,e),u&(T|N)&&(u&G?kt(a,ie):ke(a)))}}let Ar=!1;var ht,Ut,Mt;function $r(){if(ht===void 0){ht=window;var t=Element.prototype,e=Node.prototype;Ut=lt(e,"firstChild").get,Mt=lt(e,"nextSibling").get,t.__click=void 0,t.__className="",t.__attributes=null,t.__styles=null,t.__e=void 0,Text.prototype.__t=void 0}}function Gt(t=""){return document.createTextNode(t)}function we(t){return Ut.call(t)}function De(t){return Mt.call(t)}function Rt(t,e){return we(t)}function Ir(t,e){{var r=we(t);return r instanceof Comment&&r.data===""?De(r):r}}function Br(t,e=1,r=!1){let n=t;for(;e--;)n=De(n);return n}function Pr(t){t.textContent=""}function qe(t){var e=G|z,r=x!==null&&x.f&G?x:null;return w===null||r!==null&&r.f&N?e|=N:w.f|=$t,{ctx:b,deps:null,effects:null,equals:It,f:e,fn:t,reactions:null,rv:0,v:null,wv:0,parent:r??w}}function Sr(t){const e=qe(t);return e.equals=Bt,e}function Ft(t){var e=t.effects;if(e!==null){t.effects=null;for(var r=0;r<e.length;r+=1)V(e[r])}}function Dr(t){for(var e=t.parent;e!==null;){if(!(e.f&G))return e;e=e.parent}return null}function Tr(t){var e,r=w;se(Dr(t));try{Ft(t),e=Jt(t)}finally{se(r)}return e}function Ct(t){var e=Tr(t),r=(Q||t.f&N)&&t.deps!==null?ie:T;j(t,r),t.equals(e)||(t.v=e,t.wv=Kt())}function Nt(t){w===null&&x===null&&cr(),x!==null&&x.f&N&&w===null&&fr(),ze&&lr()}function kr(t,e){var r=e.last;r===null?e.last=e.first=t:(r.next=t,t.prev=r,e.last=t)}function de(t,e,r,n=!0){var s=(t&fe)!==0,i=w,a={ctx:b,deps:null,nodes_start:null,nodes_end:null,f:t|z,first:null,fn:e,last:null,next:null,parent:s?null:i,prev:null,teardown:null,transitions:null,wv:0};if(r){var u=re;try{_t(!0),Ye(a),a.f|=ar}catch(c){throw V(a),c}finally{_t(u)}}else e!==null&&ke(a);var o=r&&a.deps===null&&a.first===null&&a.nodes_start===null&&a.teardown===null&&(a.f&($t|me))===0;if(!o&&!s&&n&&(i!==null&&kr(a,i),x!==null&&x.f&G)){var f=x;(f.effects??(f.effects=[])).push(a)}return a}function Re(t){Nt();var e=w!==null&&(w.f&H)!==0&&b!==null&&!b.m;if(e){var r=b;(r.e??(r.e=[])).push({fn:t,effect:w,reaction:x})}else{var n=Lt(t);return n}}function Ur(t){return Nt(),Gr(t)}function Mr(t){const e=de(fe,t,!0);return(r={})=>new Promise(n=>{r.outro?be(e,()=>{V(e),n(void 0)}):(V(e),n(void 0))})}function Lt(t){return de(At,t,!1)}function Gr(t){return de(Be,t,!0)}function Rr(t,e=[],r=qe){const n=e.map(r);return je(()=>t(...n.map(F)))}function je(t,e=0){return de(Be|Le|e,t,!0)}function le(t,e=!0){return de(Be|H,t,!0,e)}function Wt(t){var e=t.teardown;if(e!==null){const r=ze,n=x;gt(!0),ne(null);try{e.call(null)}finally{gt(r),ne(n)}}}function qt(t,e=!1){var r=t.first;for(t.first=t.last=null;r!==null;){var n=r.next;V(r,e),r=n}}function Fr(t){for(var e=t.first;e!==null;){var r=e.next;e.f&H||V(e),e=r}}function V(t,e=!0){var r=!1;if((e||t.f&ur)&&t.nodes_start!==null){for(var n=t.nodes_start,s=t.nodes_end;n!==null;){var i=n===s?null:De(n);n.remove(),n=i}r=!0}qt(t,e&&!r),Ie(t,0),j(t,Pe);var a=t.transitions;if(a!==null)for(const o of a)o.stop();Wt(t);var u=t.parent;u!==null&&u.first!==null&&jt(t),t.next=t.prev=t.teardown=t.ctx=t.deps=t.fn=t.nodes_start=t.nodes_end=null}function jt(t){var e=t.parent,r=t.prev,n=t.next;r!==null&&(r.next=n),n!==null&&(n.prev=r),e!==null&&(e.first===t&&(e.first=n),e.last===t&&(e.last=r))}function be(t,e){var r=[];He(t,r,!0),Ht(r,()=>{V(t),e&&e()})}function Ht(t,e){var r=t.length;if(r>0){var n=()=>--r||e();for(var s of t)s.out(n)}else e()}function He(t,e,r){if(!(t.f&W)){if(t.f^=W,t.transitions!==null)for(const a of t.transitions)(a.is_global||r)&&e.push(a);for(var n=t.first;n!==null;){var s=n.next,i=(n.f&We)!==0||(n.f&H)!==0;He(n,e,i?r:!1),n=s}}}function Oe(t){zt(t,!0)}function zt(t,e){if(t.f&W){t.f^=W,t.f&T||(t.f^=T),pe(t)&&(j(t,z),ke(t));for(var r=t.first;r!==null;){var n=r.next,s=(r.f&We)!==0||(r.f&H)!==0;zt(r,s?e:!1),r=n}if(t.transitions!==null)for(const i of t.transitions)(i.is_global||e)&&i.in()}}let ge=!1,Ee=!1,Ae=null,re=!1,ze=!1;function _t(t){re=t}function gt(t){ze=t}let Fe=[],oe=0;let x=null,q=!1;function ne(t){x=t}let w=null;function se(t){w=t}let C=null;function Cr(t){C=t}let D=null,M=0,L=null;function Nr(t){L=t}let Yt=1,$e=0,Q=!1;function Kt(){return++Yt}function pe(t){var f;var e=t.f;if(e&z)return!0;if(e&ie){var r=t.deps,n=(e&N)!==0;if(r!==null){var s,i,a=(e&ye)!==0,u=n&&w!==null&&!Q,o=r.length;if(a||u){for(s=0;s<o;s++)i=r[s],(a||!((f=i==null?void 0:i.reactions)!=null&&f.includes(t)))&&(i.reactions??(i.reactions=[])).push(t);a&&(t.f^=ye)}for(s=0;s<o;s++)if(i=r[s],pe(i)&&Ct(i),i.wv>t.wv)return!0}(!n||w!==null&&!Q)&&j(t,T)}return!1}function Lr(t,e){for(var r=e;r!==null;){if(r.f&me)try{r.fn(t);return}catch{r.f^=me}r=r.parent}throw ge=!1,t}function Wr(t){return(t.f&Pe)===0&&(t.parent===null||(t.parent.f&me)===0)}function Te(t,e,r,n){if(ge){if(r===null&&(ge=!1),Wr(e))throw t;return}r!==null&&(ge=!0);{Lr(t,e);return}}function Vt(t,e,r=0){var n=t.reactions;if(n!==null)for(var s=0;s<n.length;s++){var i=n[s];i.f&G?Vt(i,e,r+1):e===i&&(r===0?j(i,z):i.f&T&&j(i,ie),ke(i))}}function Jt(t){var h;var e=D,r=M,n=L,s=x,i=Q,a=C,u=b,o=q,f=t.f;D=null,M=0,L=null,x=f&(H|fe)?null:t,Q=(f&N)!==0&&(!re||(s===null||o)&&t.parent!==null),C=null,dt(t.ctx),q=!1,$e++;try{var c=(0,t.fn)(),d=t.deps;if(D!==null){var l;if(Ie(t,M),d!==null&&M>0)for(d.length=M+D.length,l=0;l<D.length;l++)d[M+l]=D[l];else t.deps=d=D;if(!Q)for(l=M;l<d.length;l++)((h=d[l]).reactions??(h.reactions=[])).push(t)}else d!==null&&M<d.length&&(Ie(t,M),d.length=M);if(Se()&&L!==null&&!(t.f&(G|ie|z)))for(l=0;l<L.length;l++)Vt(L[l],t);return s!==null&&$e++,c}finally{D=e,M=r,L=n,x=s,Q=i,C=a,dt(u),q=o}}function qr(t,e){let r=e.reactions;if(r!==null){var n=tr.call(r,t);if(n!==-1){var s=r.length-1;s===0?r=e.reactions=null:(r[n]=r[s],r.pop())}}r===null&&e.f&G&&(D===null||!D.includes(e))&&(j(e,ie),e.f&(N|ye)||(e.f^=ye),Ft(e),Ie(e,0))}function Ie(t,e){var r=t.deps;if(r!==null)for(var n=e;n<r.length;n++)qr(t,r[n])}function Ye(t){var e=t.f;if(!(e&Pe)){j(t,T);var r=w,n=b;w=t;try{e&Le?Fr(t):qt(t),Wt(t);var s=Jt(t);t.teardown=typeof s=="function"?s:null,t.wv=Yt;var i=t.deps,a;ot&&_r&&t.f&z}catch(u){Te(u,t,r,n||t.ctx)}finally{w=r}}}function jr(){if(oe>1e3){oe=0;try{dr()}catch(t){if(Ae!==null)Te(t,Ae,null);else throw t}}oe++}function Hr(t){var e=t.length;if(e!==0){jr();var r=re;re=!0;try{for(var n=0;n<e;n++){var s=t[n];s.f&T||(s.f^=T);var i=[];Xt(s,i),zr(i)}}finally{re=r}}}function zr(t){var e=t.length;if(e!==0)for(var r=0;r<e;r++){var n=t[r];if(!(n.f&(Pe|W)))try{pe(n)&&(Ye(n),n.deps===null&&n.first===null&&n.nodes_start===null&&(n.teardown===null?jt(n):n.fn=null))}catch(s){Te(s,n,null,n.ctx)}}}function Yr(){if(Ee=!1,oe>1001)return;const t=Fe;Fe=[],Hr(t),Ee||(oe=0,Ae=null)}function ke(t){Ee||(Ee=!0,queueMicrotask(Yr)),Ae=t;for(var e=t;e.parent!==null;){e=e.parent;var r=e.f;if(r&(fe|H)){if(!(r&T))return;e.f^=T}}Fe.push(e)}function Xt(t,e){var r=t.first,n=[];e:for(;r!==null;){var s=r.f,i=(s&H)!==0,a=i&&(s&T)!==0,u=r.next;if(!a&&!(s&W))if(s&Be){if(i)r.f^=T;else{var o=x;try{x=r,pe(r)&&Ye(r)}catch(l){Te(l,r,null,r.ctx)}finally{x=o}}var f=r.first;if(f!==null){r=f;continue}}else s&At&&n.push(r);if(u===null){let l=r.parent;for(;l!==null;){if(t===l)break e;var c=l.next;if(c!==null){r=c;continue e}l=l.parent}}r=u}for(var d=0;d<n.length;d++)f=n[d],e.push(f),Xt(f,e)}function F(t){var e=t.f,r=(e&G)!==0;if(x!==null&&!q){C!==null&&C.includes(t)&&pr();var n=x.deps;t.rv<$e&&(t.rv=$e,D===null&&n!==null&&n[M]===t?M++:D===null?D=[t]:D.push(t))}else if(r&&t.deps===null&&t.effects===null){var s=t,i=s.parent;i!==null&&!(i.f&N)&&(s.f^=N)}return r&&(s=t,pe(s)&&Ct(s)),t.v}function Qt(t){var e=q;try{return q=!0,t()}finally{q=e}}const Kr=-7169;function j(t,e){t.f=t.f&Kr|e}function Vr(t){if(!(typeof t!="object"||!t||t instanceof EventTarget)){if(ct in t)Ce(t);else if(!Array.isArray(t))for(let e in t){const r=t[e];typeof r=="object"&&r&&ct in r&&Ce(r)}}}function Ce(t,e=new Set){if(typeof t=="object"&&t!==null&&!(t instanceof EventTarget)&&!e.has(t)){e.add(t),t instanceof Date&&t.getTime();for(let n in t)try{Ce(t[n],e)}catch{}const r=sr(t);if(r!==Object.prototype&&r!==Array.prototype&&r!==Map.prototype&&r!==Set.prototype&&r!==Date.prototype){const n=nr(r);for(let s in n){const i=n[s].get;if(i)try{i.call(t)}catch{}}}}}const Jr=["touchstart","touchmove"];function Xr(t){return Jr.includes(t)}const Qr=new Set,vt=new Set;function _e(t){var y;var e=this,r=e.ownerDocument,n=t.type,s=((y=t.composedPath)==null?void 0:y.call(t))||[],i=s[0]||t.target,a=0,u=t.__root;if(u){var o=s.indexOf(u);if(o!==-1&&(e===document||e===window)){t.__root=e;return}var f=s.indexOf(e);if(f===-1)return;o<=f&&(a=o)}if(i=s[a]||t.target,i!==e){rr(t,"currentTarget",{configurable:!0,get(){return i||r}});var c=x,d=w;ne(null),se(null);try{for(var l,h=[];i!==null;){var m=i.assignedSlot||i.parentNode||i.host||null;try{var B=i["__"+n];if(B!==void 0&&!i.disabled)if(Et(B)){var[g,...p]=B;g.apply(i,[t,...p])}else B.call(i,t)}catch(E){l?h.push(E):l=E}if(t.cancelBubble||m===e||m===null)break;i=m}if(l){for(let E of h)queueMicrotask(()=>{throw E});throw l}}finally{t.__root=e,delete t.currentTarget,ne(c),se(d)}}}function Zr(t){var e=document.createElement("template");return e.innerHTML=t,e.content}function mt(t,e){var r=w;r.nodes_start===null&&(r.nodes_start=t,r.nodes_end=e)}function Ue(t,e){var r=(e&xr)!==0,n=(e&wr)!==0,s,i=!t.startsWith("<!>");return()=>{s===void 0&&(s=Zr(i?t:"<!>"+t),r||(s=we(s)));var a=n?document.importNode(s,!0):s.cloneNode(!0);if(r){var u=we(a),o=a.lastChild;mt(u,o)}else mt(a,a);return a}}function ve(t,e){t!==null&&t.before(e)}function en(t,e){var r=e==null?"":typeof e=="object"?e+"":e;r!==(t.__t??(t.__t=t.nodeValue))&&(t.__t=r,t.nodeValue=r+"")}function tn(t,e){return rn(t,e)}const ee=new Map;function rn(t,{target:e,anchor:r,props:n={},events:s,context:i,intro:a=!0}){$r();var u=new Set,o=d=>{for(var l=0;l<d.length;l++){var h=d[l];if(!u.has(h)){u.add(h);var m=Xr(h);e.addEventListener(h,_e,{passive:m});var B=ee.get(h);B===void 0?(document.addEventListener(h,_e,{passive:m}),ee.set(h,1)):ee.set(h,B+1)}}};o(Ne(Qr)),vt.add(o);var f=void 0,c=Mr(()=>{var d=r??e.appendChild(Gt());return le(()=>{if(i){Pt({});var l=b;l.c=i}s&&(n.$$events=s),f=t(d,n)||{},i&&St()}),()=>{var m;for(var l of u){e.removeEventListener(l,_e);var h=ee.get(l);--h===0?(document.removeEventListener(l,_e),ee.delete(l)):ee.set(l,h)}vt.delete(o),d!==r&&((m=d.parentNode)==null||m.removeChild(d))}});return nn.set(f,c),f}let nn=new WeakMap;function Zt(t,e,r=!1){var n=t,s=null,i=null,a=br,u=r?We:0,o=!1;const f=(d,l=!0)=>{o=!0,c(l,d)},c=(d,l)=>{a!==(a=d)&&(a?(s?Oe(s):l&&(s=le(()=>l(n))),i&&be(i,()=>{i=null})):(i?Oe(i):l&&(i=le(()=>l(n))),s&&be(s,()=>{s=null})))};je(()=>{o=!1,e(f),o||c(null,null)},u)}function sn(t,e){return e}function an(t,e,r,n){for(var s=[],i=e.length,a=0;a<i;a++)He(e[a].e,s,!0);var u=i>0&&s.length===0&&r!==null;if(u){var o=r.parentNode;Pr(o),o.append(r),n.clear(),Y(t,e[0].prev,e[i-1].next)}Ht(s,()=>{for(var f=0;f<i;f++){var c=e[f];u||(n.delete(c.k),Y(t,c.prev,c.next)),V(c.e,!u)}})}function un(t,e,r,n,s,i=null){var a=t,u={flags:e,items:new Map,first:null};{var o=t;a=o.appendChild(Gt())}var f=null,c=!1,d=Sr(()=>{var l=r();return Et(l)?l:l==null?[]:Ne(l)});je(()=>{var l=F(d),h=l.length;c&&h===0||(c=h===0,on(l,u,a,s,e,n,r),i!==null&&(h===0?f?Oe(f):f=le(()=>i(a)):f!==null&&be(f,()=>{f=null})),F(d))})}function on(t,e,r,n,s,i,a){var u=t.length,o=e.items,f=e.first,c=f,d,l=null,h=[],m=[],B,g,p,y;for(y=0;y<u;y+=1){if(B=t[y],g=i(B,y),p=o.get(g),p===void 0){var E=c?c.e.nodes_start:r;l=fn(E,e,l,l===null?e.first:l.next,B,g,y,n,s,a),o.set(g,l),h=[],m=[],c=l.next;continue}if(ln(p,B,y),p.e.f&W&&Oe(p.e),p!==c){if(d!==void 0&&d.has(p)){if(h.length<m.length){var P=m[0],$;l=P.prev;var R=h[0],Z=h[h.length-1];for($=0;$<h.length;$+=1)yt(h[$],P,r);for($=0;$<m.length;$+=1)d.delete(m[$]);Y(e,R.prev,Z.next),Y(e,l,R),Y(e,Z,P),c=P,l=Z,y-=1,h=[],m=[]}else d.delete(p),yt(p,c,r),Y(e,p.prev,p.next),Y(e,p,l===null?e.first:l.next),Y(e,l,p),l=p;continue}for(h=[],m=[];c!==null&&c.k!==g;)c.e.f&W||(d??(d=new Set)).add(c),m.push(c),c=c.next;if(c===null)continue;p=c}h.push(p),l=p,c=p.next}if(c!==null||d!==void 0){for(var J=d===void 0?[]:Ne(d);c!==null;)c.e.f&W||J.push(c),c=c.next;var k=J.length;if(k>0){var U=u===0?r:null;an(e,J,U,o)}}w.first=e.first&&e.first.e,w.last=l&&l.e}function ln(t,e,r,n){Tt(t.v,e),t.i=r}function fn(t,e,r,n,s,i,a,u,o,f){var c=(o&vr)!==0,d=(o&yr)===0,l=c?d?Dt(s):xe(s):s,h=o&mr?xe(a):a,m={i:h,v:l,k:i,a:null,e:null,prev:r,next:n};try{return m.e=le(()=>u(t,l,h,f),Ar),m.e.prev=r&&r.e,m.e.next=n&&n.e,r===null?e.first=m:(r.next=m,r.e.next=m.e),n!==null&&(n.prev=m,n.e.prev=m.e),m}finally{}}function yt(t,e,r){for(var n=t.next?t.next.e.nodes_start:r,s=e?e.e.nodes_start:r,i=t.e.nodes_start;i!==n;){var a=De(i);s.before(i),i=a}}function Y(t,e,r){e===null?t.first=r:(e.next=r,e.e.next=r&&r.e),r!==null&&(r.prev=e,r.e.prev=e&&e.e)}function cn(t=!1){const e=b,r=e.l.u;if(!r)return;let n=()=>Vr(e.s);if(t){let s=0,i={};const a=qe(()=>{let u=!1;const o=e.s;for(const f in o)o[f]!==i[f]&&(i[f]=o[f],u=!0);return u&&s++,s});n=()=>F(a)}r.b.length&&Ur(()=>{xt(e,n),ft(r.b)}),Re(()=>{const s=Qt(()=>r.m.map(ir));return()=>{for(const i of s)typeof i=="function"&&i()}}),r.a.length&&Re(()=>{xt(e,n),ft(r.a)})}function xt(t,e){if(t.l.s)for(const r of t.l.s)F(r);e()}function dn(t){b===null&&Or(),ce&&b.l!==null?pn(b).m.push(t):Re(()=>{const e=Qt(t);if(typeof e=="function")return e})}function pn(t){var e=t.l;return e.u??(e.u={a:[],b:[],m:[]})}const hn="5";typeof window<"u"&&(window.__svelte||(window.__svelte={v:new Set})).v.add(hn);gr();const te={f32:Float32Array,u32:Uint32Array,i32:Int32Array},er=Uint32Array,_n=Uint32Array;function S(t){let e=1;for(const r of t)e*=r;return e}function gn(t=0,e=1){const r=1-Math.random(),n=Math.random();return Math.sqrt(-2*Math.log(r))*Math.cos(2*Math.PI*n)*e+t}function Ge(t){let e=new Array(t.length);e[e.length-1]=1;for(let r=t.length-1;r>0;r--)e[r-1]=e[r]*t[r];return e}function I(t,e){return Math.ceil(t/e)}function X(t){const e=new t.constructor(t.length);for(let r=0;r<t.length;r++)e[r]=t[r];return e}function wt(t,e){const r=X(t);for(let n=0;n<e.length;n++){const s=e[n];t[n]=r[s]}}function vn(t,e,r){const n=t[e];t[e]=r,t[r]=n}function ue(t,e){if(t.length!==e.length)return!1;for(let r=0;r<t.length;r++)if(t[r]!==e[r])return!1;return!0}function mn(t,e,r,n=1/0){let s="";function i(a,u){for(let o=0;o<e[a];o++){if(o>n){s+="...";return}const f=u+o*r[a];if(a===e.length-1)s+=`${t[f]}`,o<e[a]-1&&(s+=", ");else{if(o>0)for(let c=0;c<a+1;c++)s+=" ";if(s+="[",i(a+1,f),s+="]",o<e[a]-1){s+=",";for(let c=0;c<e.length-a-1;c++)s+=`
`}}}}return s+="[",i(0,0),s+="]",s}function ae(t,e){return e<0?t+=e:e}function A(t,e,r="gid.x",n=-1,s=""){let i="",a=1;for(let u=t.length-1+n+1;u>=0;u--){const o=`(${s}(${r}/${a})%${t[u]})`;a*=t[u];const f=e[u];i+=`(${f}*${o})`,u>0&&(i+="+")}return i}function bt(t,e,r){const n=new t.constructor(t.length+1);n[e]=r;let s=0;for(let i=0;i<n.length;i++)i!==e&&(n[i]=t[s],s++);return n}let O;class _{constructor(e,r,n,s,i=!0){v(O,"GPU must exist. Tensor.setDevice() once!"),this.gpuBuffer=e,this.shape=er.from(r),this.strides=_n.from(n),this.dtype=s,this.owned=i}get DTypedArray(){return te[this.dtype]}static setDevice(e){v(e!==void 0,"device not found!"),O=new Ke(e)}static get gpu(){return v(O!==void 0,"gpu not found!"),O}setGPUBuffer(e,r=void 0){r=r===void 0?this.shape:r,e=this.DTypedArray.from(e),S(r)!==S(this.shape)&&(this.gpuBuffer.free(),this.gpuBuffer=O.memAlloc(e.byteLength)),O.memcpyHostToDevice(this.gpuBuffer,e)}static fill(e,r,n="f32"){const s=S(r),i=O.memAlloc(s*te[n].BYTES_PER_ELEMENT),a=256;return O.SourceModule(`
			@group(0) @binding(0) var<storage, read_write> data: array<${n}>;
			@compute @workgroup_size(${a})
			fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${s}) {
					data[gid.x] = ${n}(${e});
				}
			}`).getFunction("main")([I(s,a)],i),new _(i,r,Ge(r),n)}static tensor(e,r,n="f32"){const s=new te[n](e),i=O.memAlloc(s.byteLength);return O.memcpyHostToDevice(i,s),new _(i,r,Ge(r),n)}static empty(e,r="f32"){const n=O.memAlloc(S(e)*te[r].BYTES_PER_ELEMENT);return new _(n,e,Ge(e),r)}static random(e,r="f32"){const n=new te[r](S(e)).fill(0).map(s=>Math.random());return _.tensor(n,e,r)}static randn(e,r=0,n=1,s="f32"){const i=new te[s](S(e)).fill(0).map(a=>gn(r,n));return _.tensor(i,e,s)}static _elementWiseUnaryOpInplace(e,r){const n=S(e.shape),s=256,i=e.dtype;O.SourceModule(`
			@group(0) @binding(0) var<storage, read_write> dst: array<${i}>;
			@compute @workgroup_size(${s})
			fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${n}) {
					let dstIdx = ${A(e.shape,e.strides,"gid.x",-1)};
					${r}
				}
			}
			`).getFunction("main")([I(n,s)],e.gpuBuffer)}_elementWiseUnaryOpInplace(e){return _._elementWiseUnaryOpInplace(this,e),this}static _elementWiseUnaryOp(e,r,n){v(ue(e.shape,r.shape),"dst must have shape as src");const s=S(e.shape),i=256,a=e.dtype;O.SourceModule(`
			@group(0) @binding(0) var<storage, read_write> dst: array<${a}>;
			@group(0) @binding(1) var<storage, read> src: array<${a}>;
			@compute @workgroup_size(${i})
			fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${s}) {
					let srcIdx = ${A(r.shape,r.strides,"gid.x",-1)};
					let dstIdx = ${A(e.shape,e.strides,"gid.x",-1)};
					${n}
				}
			}
			`).getFunction("main")([I(s,i)],e.gpuBuffer,r.gpuBuffer)}_elementWiseUnaryOp(e){const r=_.empty(this.shape,this.dtype);return _._elementWiseUnaryOp(r,this,e),r}static sumLastDimension(e,r){v(e.dtype===r.dtype,"dst and src dtypes must match"),v(e.shape.at(-1)===1,"dimension we sum over should be 1 in dst");const n=S(e.shape),s=256,i=e.dtype;return O.SourceModule(`
			@group(0) @binding(0) var<storage, read_write> dst: array<${i}>;
			@group(0) @binding(1) var<storage, read> src: array<${i}>;

			@compute @workgroup_size(${s})
			fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${n}) {
					let baseSrcIdx = ${A(r.shape,r.strides,"gid.x",-2)};
					let baseDstIdx = ${A(e.shape,e.strides,"gid.x",-2)};
					var summed = ${i}(0);
					for(var i: u32 = 0; i < ${r.shape.at(-1)}; i++) {
						summed += src[baseSrcIdx + i*${r.strides.at(-1)}];
					}
					dst[baseDstIdx] = summed;
				}
			}`).getFunction("main")([I(n,s)],e.gpuBuffer,r.gpuBuffer),e}static reduceAnyDimensionGivenLastDimensionFunc(e,r,n,s=-1){const i=new Uint8Array(n.shape.length).fill(0).map((u,o)=>o);s=ae(n.shape.length,s);const a=n.shape.length-1;vn(i,s,a),e(r.transpose(i),n.transpose(i))}static sum(e,r,n=-1){_.reduceAnyDimensionGivenLastDimensionFunc(_.sumLastDimension,e,r,n)}static _elementWiseBinaryOp(e,r,n,s){v(ue(r.shape,n.shape),"srcA and srcB must have the same shape"),v(ue(e.shape,n.shape),"dst, srcA, and srcB, must have the same shape");const i=S(e.shape),a=256,u=e.dtype;O.SourceModule(`
			@group(0) @binding(0) var<storage, read_write> dst: array<${u}>;
			@group(0) @binding(1) var<storage, read> srcA: array<${u}>;
			@group(0) @binding(2) var<storage, read> srcB: array<${u}>;

		 	@compute @workgroup_size(${a})
		 	fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${i}) {
					let dstIdx = ${A(e.shape,e.strides,"gid.x",-1)};
					let srcAIdx = ${A(r.shape,r.strides,"gid.x",-1)};
					let srcBIdx = ${A(n.shape,n.strides,"gid.x",-1)};
					${s}
				}
			}`).getFunction("main")([I(i,a)],e.gpuBuffer,r.gpuBuffer,n.gpuBuffer)}_elementWiseBinaryOp(e,r){let n=!1;typeof e=="number"&&(e=_.tensor([e],this.shape,this.dtype),n=!0);const s=_.empty(e.shape,e.dtype);return _._elementWiseBinaryOp(s,this,e,r),n&&e.free(),s}static _elementWiseBinaryOpInplace(e,r,n){v(ue(e.shape,r.shape),"dst, src, must have the same shape");const s=S(e.shape),i=256,a=e.dtype;O.SourceModule(`
			@group(0) @binding(0) var<storage, read_write> dst: array<${a}>;
			@group(0) @binding(1) var<storage, read> src: array<${a}>;

		 	@compute @workgroup_size(${i})
		 	fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${s}) {
					let dstIdx = ${A(e.shape,e.strides,"gid.x",-1)};
					let srcIdx = ${A(r.shape,r.strides,"gid.x",-1)};
					${n}
				}
			}`).getFunction("main")([I(s,i)],e.gpuBuffer,r.gpuBuffer)}add(e){return this._elementWiseBinaryOp(e,"dst[dstIdx] = srcA[srcAIdx]+srcB[srcBIdx];")}sub(e){return this._elementWiseBinaryOp(e,"dst[dstIdx] = srcA[srcAIdx]-srcB[srcBIdx];")}mul(e){return this._elementWiseBinaryOp(e,"dst[dstIdx] = srcA[srcAIdx]*srcB[srcBIdx];")}div(e){return this._elementWiseBinaryOp(e,"dst[dstIdx] = srcA[srcAIdx]/srcB[srcBIdx];")}pow(e){return this._elementWiseBinaryOp(e,`dst[dstIdx] = ${this.dtype}(pow(f32(srcA[srcAIdx]), f32(srcB[srcBIdx])));`)}add_(e){return _._elementWiseBinaryOpInplace(this,e,"dst[dstIdx] += src[srcIdx];"),this}sub_(e){return _._elementWiseBinaryOpInplace(this,e,"dst[dstIdx] -= src[srcIdx];"),this}sum(e=-1){const r=X(this.shape);r[ae(r.length,e)]=1;const n=_.empty(r,this.dtype);return _.sum(n,this,e),n}static maxLastDim(e,r){v(e.dtype===r.dtype,"dst and src dtypes must match"),v(e.shape.at(-1)===1,"dimension we sum over should be 1 in dst");const n=S(e.shape),s=256,i=e.dtype;return O.SourceModule(`
			@group(0) @binding(0) var<storage, read_write> dst: array<${i}>;
			@group(0) @binding(1) var<storage, read> src: array<${i}>;

			@compute @workgroup_size(${s})
			fn main(@builtin(global_invocation_id) gid : vec3u) {
				if(gid.x < ${n}) {
					let baseSrcIdx = ${A(r.shape,r.strides,"gid.x",-2)};
					let baseDstIdx = ${A(e.shape,e.strides,"gid.x",-2)};
					var max: ${i} = src[baseSrcIdx];
					for(var i: u32 = 0; i < ${r.shape.at(-1)}; i++) {
						let cur = src[baseSrcIdx + i*${r.strides.at(-1)}];
						if(cur > max) {
							max = cur;
						}
					}
					dst[baseDstIdx] = max;
				}
			}`).getFunction("main")([I(n,s)],e.gpuBuffer,r.gpuBuffer),e}static max(e,r,n=-1){_.reduceAnyDimensionGivenLastDimensionFunc(_.maxLastDim,e,r,n)}max(e=-1){const r=X(this.shape);r[ae(r.length,e)]=1;const n=_.empty(r,this.dtype);return _.max(n,this,e),n}softmax(e=-1){const r=this.max(e),n=this.sub(r.expand(this.shape)),s=n.exp(),i=s.sum(e),a=s.div(i.expand(s.shape));return r.free(),n.free(),i.free(),s.free(),a}fillInplace(e){return this._elementWiseUnaryOpInplace(`dst[dstIdx] = ${this.dtype}(${e});`)}contiguous(){return this._elementWiseUnaryOp(dst,src,"dst[dstIdx] = src[srcIdx];")}exp(){return this._elementWiseUnaryOp("dst[dstIdx] = exp(src[srcIdx]);")}relu(){return this._elementWiseUnaryOp("dst[dstIdx] = max(src[srcIdx], 0);")}reciprocal(){return this._elementWiseUnaryOp(other,"dst[dstIdx] = 1/src[srcIdx];")}log(e=1e-6){return this._elementWiseUnaryOp(`dst[dstIdx] = log(src[srcIdx] + ${e});`)}get T(){return this.transpose()}transpose(e=void 0){if(this.shape===1)return this;e===void 0&&(e=[this.shape.length-1,0]);const r=X(this.shape),n=X(this.strides);return wt(r,e),wt(n,e),new _(this.gpuBuffer,r,n,this.dtype,!1)}unsqueeze(e=0){e=ae(this.shape.length+1,e);const r=bt(this.shape,e,1),n=this.shape.slice(e).reduce((i,a)=>i*a,1),s=bt(this.strides,e,n);return new _(this.gpuBuffer,r,s,this.dtype,!1)}expandTo(e,r){if(r=ae(this.shape.length,r),e===this.shape[r])return this;const n=X(this.shape),s=X(this.strides);return n[r]=e,s[r]=0,new _(this.gpuBuffer,n,s,this.dtype,!1)}expand(e){v(e.length===this.shape.length,"Must have same number of dims");let r=this;for(let n=0;n<e.length;n++){const s=e[n];r=r.expandTo(s,n)}return r}static matmul(e,r,n){v(r.shape.length===2&&n.shape.length===2&&e.shape.length===2,"tensors are matrix shaped"),v(r.shape.at(-1)===n.shape.at(0),"Inner dimension must be the same"),v(e.shape[0]===r.shape[0]&&e.shape[1]===n.shape[1],"output dimension lines up");const s=r.shape[1],i=e.dtype,a=16,u=16,o=O.SourceModule(`
			@group(0) @binding(0) var<storage, read_write> dst: array<${i}>;
			@group(0) @binding(1) var<storage, read> srcA: array<${i}>;
			@group(0) @binding(2) var<storage, read> srcB: array<${i}>;

		 	@compute @workgroup_size(${a}, ${u})
		 	fn main(@builtin(global_invocation_id) gid : vec3u) {
				let i = gid.x;
				let j = gid.y;
				if(gid.x < ${e.shape[0]} && gid.y < ${e.shape[1]}) {
					var summed: ${i} = 0;
					for(var k: u32 = 0; k < ${s}; k++) {
						let srcAIdx = i*${r.strides[0]} + k*${r.strides[1]};
						let srcBIdx = k*${n.strides[0]} + j*${n.strides[1]};
						summed += srcA[srcAIdx]*srcB[srcBIdx];	
					}

					let dstIdx = i*${e.strides[0]} + j*${e.strides[1]};
					dst[dstIdx] = summed;
				}
			}
		`).getFunction("main"),f=[I(e.shape[0],a),I(e.shape[1],u)];o(f,e.gpuBuffer,r.gpuBuffer,n.gpuBuffer)}matmul(e){const r=_.empty([this.shape[0],e.shape[1]],this.dtype);return _.matmul(r,this,e),r}static bmm(e,r,n){v(r.shape.at(-1)===n.shape.at(-2),"Inner dimension must be the same"),v(e.shape.at(-2)===r.shape.at(-2)&&e.shape.at(-1)===n.shape.at(-1),"output dimension lines up");const s=S(e.shape.slice(0,-2)),i=e.shape.at(-2),a=e.shape.at(-1),u=r.shape.at(-1),o=e.dtype,f=4,c=8,d=8,l=O.SourceModule(`
			@group(0) @binding(0) var<storage, read_write> dst: array<${o}>;
			@group(0) @binding(1) var<storage, read> srcA: array<${o}>;
			@group(0) @binding(2) var<storage, read> srcB: array<${o}>;

		 	@compute @workgroup_size(${f}, ${c}, ${d})
		 	fn main(@builtin(global_invocation_id) gid : vec3u) {
				let b = gid.x;
				let i = gid.y;
				let j = gid.z;
				if(b < ${s} && i < ${i} && j < ${a}) {
					let srcAOffset = ${A(r.shape,r.strides,"b",-3)};
					let srcBOffset = ${A(n.shape,n.strides,"b",-3)};
					let dstOffset = ${A(e.shape,e.strides,"b",-3)};

					var summed: ${o} = 0;
					for(var k: u32 = 0; k < ${u}; k++) {
						let srcAIdx = srcAOffset + i*${r.strides.at(-2)} + k*${r.strides.at(-1)};
						let srcBIdx = srcBOffset + k*${n.strides.at(-2)} + j*${n.strides.at(-1)};
						summed += srcA[srcAIdx]*srcB[srcBIdx];	
					}

					let dstIdx = dstOffset + i*${e.strides.at(-2)} + j*${e.strides.at(-1)};
					dst[dstIdx] = summed;
				}
			}
		`).getFunction("main"),h=[I(s,f),I(i,c),I(a,d)];l(h,e.gpuBuffer,r.gpuBuffer,n.gpuBuffer)}bmm(e){const r=[...this.shape.slice(0,-2),this.shape.at(-2),e.shape.at(-1)],n=_.empty(r);return _.bmm(n,this,e),n}static _softmaxJacobianLastDim(e,r){v(e.shape.at(-1)===e.shape.at(-2)&&e.shape.at(-1)===r.shape.at(-1));const n=S(e.shape.slice(-2)),s=e.shape.at(-1),i=e.dtype,a=4,u=8,o=8,f=O.SourceModule(`
			@group(0) @binding(0) var<storage, read_write> dst: array<${i}>;
			@group(0) @binding(1) var<storage, read> s: array<${i}>;

		 	@compute @workgroup_size(${a}, ${u}, ${o})
		 	fn main(@builtin(global_invocation_id) gid : vec3u) {
				let b = gid.x;
				let i = gid.y;
				let j = gid.z;

				if(b < ${n} && i < ${s} && j < ${s}) {
					let dstOffset = ${A(e.shape,e.strides,"b",-3)};
					let sOffset = ${A(r.shape,r.strides,"b",-2)};

					let siIdx = sOffset + i*${r.strides.at(-1)};
					let sjIdx = sOffset + j*${r.strides.at(-1)};
					let dijIdx = dstOffset + i*${e.strides.at(-2)} + j*${e.strides.at(-1)};

					let si = s[siIdx];
					let sj = s[sjIdx];
					if(i!=j) {
						dst[dijIdx] = -si*sj;
					}
					else {
						dst[dijIdx] = si*(1-si);
					}
				}
			}
		`).getFunction("main"),c=[I(n,a),I(s,u),I(s,o)];f(c,e.gpuBuffer,r.gpuBuffer)}_softmaxJacobian(){const e=this.shape.at(-1),r=_.empty([...this.shape.slice(0,-1),e,e]);return _._softmaxJacobianLastDim(r,this),r}async print(e=!0){this.assertNotFreed();const r=await this.cpuBuffer();let n="";n+=`dtype='${this.dtype}', `,n+=`shape=[${this.shape}], `,n+=`strides=[${this.strides}],
`,n+=`gpuBuffer=
${mn(r,this.shape,this.strides,e?8:1/0)}
`,console.log(n)}async cpuBuffer(){return O.mapGPUToCPU(this.gpuBuffer,this.DTypedArray)}free(){this.gpuBuffer===void 0&&console.warn("Tried to free a gpuBuffer twice!"),this.gpuBuffer&&this.owned&&O.free(this.gpuBuffer),this.gpuBuffer=void 0}assertNotFreed(){v(this.gpuBuffer!==void 0,"This GPU Buffer has already been freed.")}}function v(t,e="ASSERT FAILED"){if(!t)throw new Error(e)}class Ke{constructor(e){this.device=e}static async init(){const e=await navigator.gpu.requestAdapter();v(e,"adapter exists");const r=await e.requestDevice();return v(r,"device exists"),new Ke(r)}async deviceSynchronize(){await this.device.queue.onSubmittedWorkDone()}memAlloc(e,r=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC){return v(e>0),this.device.createBuffer({size:e,usage:r})}memcpyHostToDevice(e,r){this.device.queue.writeBuffer(e,0,r,0)}async memcpyDeviceToHost(e,r){e.set(await this.mapGPUToCPU(r,e.constructor))}free(e){e.destroy()}async printGPUBuffer(e,r=Float32Array){const n=await this.mapGPUToCPU(e,r);console.log(Array.from(n),n.constructor.name)}printDeviceInfo(){console.table(this.device.adapterInfo)}async mapGPUToCPU(e,r=Float32Array){const n=this.memAlloc(e.size,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),s=this.device.createCommandEncoder();return s.copyBufferToBuffer(e,0,n,0,e.size),this.device.queue.submit([s.finish()]),await n.mapAsync(GPUMapMode.READ),new r(n.getMappedRange())}SourceModule(e){return new yn(this,e)}}class yn{constructor(e,r){this.gpu=e,this.device=e.device,this.kernel=r}getFunctionExplicitBindings(e){const r=this.device.createShaderModule({code:this.kernel}),n=this.device.createComputePipeline({layout:"auto",compute:{module:r,entryPoint:e}}),s=n.getBindGroupLayout(0);return(i,...a)=>{v(i!==void 0);const u=this.device.createBindGroup({layout:s,entries:a}),o=this.device.createCommandEncoder(),f=o.beginComputePass();f.setPipeline(n),f.setBindGroup(0,u),f.dispatchWorkgroups(...i),f.end(),this.device.queue.submit([o.finish()])}}getFunctionOnlyBuffers(e){const r=this.getFunctionExplicitBindings(e);return(n,...s)=>{const i=s.map((a,u)=>({binding:u,resource:{buffer:a}}));r(n,...i)}}getFunction(e,r=!1){return r?this.getFunctionExplicitBindings(e):this.getFunctionOnlyBuffers(e)}}const xn=14,[Ot,Ve,Je,Xe,Qe,Ze,et,tt,rt,nt,st,it,at,ut]=new Array(xn).fill(0).map((t,e)=>e),wn={[Je]:([t],e)=>t.sum(e),[Qe]:([t])=>t.pow(2),[tt]:([t],e)=>t.transpose(e),[rt]:([t],e)=>t.expand(e),[nt]:([t])=>t.relu(),[it]:([t])=>t.exp(),[at]:([t],e)=>t.softmax(e),[ut]:([t])=>t.log()},bn={[Je]:([t],e,r)=>[()=>r.expand(t.shape)],[Qe]:([t],e,r)=>[()=>{const s=t.mul(2),i=r.mul(s);return s.free(),i}],[tt]:([t],e,r,n)=>[()=>r.transpose(n)],[rt]:([t],e,r,n)=>[()=>r.expand(t.shape)],[nt]:([t],e,r)=>[()=>e._elementWiseBinaryOp(r,`
				let result = srcA[srcAIdx];
				let resultGrad = srcB[srcBIdx];
				if(result > 0) {
					dst[dstIdx] = resultGrad;
				} else {
					dst[dstIdx] = 0;
				}
				`)],[it]:([t],e,r)=>[()=>r.mul(e)],[at]:([t],e,r)=>[()=>{const s=e._softmaxJacobian(),i=r.unsqueeze(1).bmm(s);return i.shape=e.shape,i.strides=e.strides,i}],[ut]:([t],e,r)=>[()=>{const s=t.pow(-1),i=r.mul(s);return s.free(),i}]},On={[Ve]:([t,e])=>t.add(e),[Xe]:([t,e])=>t.sub(e),[Ze]:([t,e])=>t.mul(e),[et]:([t,e])=>t.matmul(e),[st]:([t,e])=>t.div(e)},En={[Ve]:([t,e],r,n)=>[()=>n,()=>n],[Xe]:([t,e],r,n)=>[()=>n,()=>n.mul(-1)],[Ze]:([t,e],r,n)=>[()=>n.mul(e),()=>n.mul(t)],[et]:([t,e],r,n)=>[()=>n.matmul(e.T),()=>t.T.matmul(n)],[st]:([t,e],r,n)=>[()=>{const a=typeof e=="number"?_.tensor([1/e],t.shape):e.pow(-1),u=n.mul(a);return a.free(),u},()=>{const a=typeof e=="number"?_.tensor([1/(e*e)],t.shape):e.pow(-2),u=t.mul(-1),o=u.mul(a),f=n.mul(a);return a.free(),o.free(),u.free(),f}]};class K{constructor(e,r,n=[],s=void 0,i=!1){this.childArgs=r,this.opArgs=n,this.OP_CODE=e,this.tensor=s,this.grad=void 0,this.requiresGrad=i}static tensor(e,r=!1){return new K(Ot,[],[],e,r,e.shape)}get shape(){return v(this.tensor,"Tensor must exist to get shape. TODO: implement shape tracker to get around this."),this.tensor.shape}_unaryOp(e,...r){return new K(e,[this],r,void 0,this.requiresGrad)}softmax(e=-1){return this._unaryOp(at,e)}exp(){return this._unaryOp(it)}sum(e){return this._unaryOp(Je,e)}square(){return this._unaryOp(Qe)}get T(){return this.transpose()}transpose(e=void 0){return this._unaryOp(tt,e)}expand(e){return this._unaryOp(rt,e)}relu(){return this._unaryOp(nt)}log(){return this._unaryOp(ut)}_binaryOp(e,r,...n){return new K(r,[this,e],n,void 0,this.requiresGrad||e.requiresGrad)}div(e){return this._binaryOp(e,st)}add(e){return this._binaryOp(e,Ve)}sub(e){return this._binaryOp(e,Xe)}mul(e){return this._binaryOp(e,Ze)}matmul(e){return this._binaryOp(e,et)}_getOpFunc(){let e;if(this.childArgs.length===1)e=wn;else if(this.childArgs.length===2)e=On;else throw new Error("Unknown op length");return v(this.OP_CODE in e,"Must have the function in the OP_MAP"),e[this.OP_CODE]}_getBackwardsOpFunc(){let e;if(this.childArgs.length===1)e=bn;else if(this.childArgs.length===2)e=En;else throw new Error("Unknown op length");return v(this.OP_CODE in e,"Must have the function in the OP_MAP"),e[this.OP_CODE]}forward(){if(this.OP_CODE===Ot)return this.tensor;let e=new Array(this.childArgs.length);for(let n=0;n<this.childArgs.length;n++){const s=this.childArgs[n];typeof s=="number"?e[n]=s:e[n]=s.forward()}const r=this._getOpFunc();return this.tensor=r(e,...this.opArgs),this.tensor}_accumulateGradient(e){this.grad===void 0&&(this.grad=_.fill(0,this.tensor.shape,this.tensor.dtype)),this.grad.add_(e)}backward(){v(this.tensor,"result needs to be evaluated"),v(ue(new er(this.tensor.shape.length).fill(1),this.tensor.shape),"Needs to be called on a reduce value (shape dimensions all 1)");const e=n=>{if(v(n.tensor,"result needs to be evaluated"),n.childArgs.length===0)return;const s=n._getBackwardsOpFunc(),i=n.childArgs.map(u=>typeof u=="number"?u:u.tensor),a=s(i,n.tensor,n.grad,...n.opArgs);for(let u=0;u<n.childArgs.length;u++){const o=n.childArgs[u],f=a[u];if(typeof o=="number"||o.requiresGrad===!1)continue;const c=f();o._accumulateGradient(c),e(o)}},r=_.fill(1,this.tensor.shape,this.tensor.dtype);this._accumulateGradient(r),e(this)}zeroGrad(){if(!(!this.grad||!this.requiresGrad)){this.grad.fillInplace(0);for(let e=0;e<this.childArgs.length;e++)typeof this.childArgs[e]!="number"&&this.childArgs[e].zeroGrad()}}freeGraph(){this.grad&&(this.grad.free(),this.grad=void 0),this.tensor&&(this.tensor.free(),this.tensor=void 0);for(let e=0;e<this.childArgs.length;e++)typeof this.childArgs[e]!="number"&&this.childArgs[e].freeGraph()}async print(...e){v(this.tensor,"result must be populated to print"),await this.tensor.print(...e)}}class An{constructor(e,r=.001){e.forEach(n=>v(n.requiresGrad,"Parameters must require gradients")),this.params=e,this.lr=r}update(){for(const e of this.params){v(e.grad&&e.tensor,"Can update data with gradient.");const r=e.grad.mul(this.lr);e.tensor.sub_(r),r.free()}}}var $n=Ue("<div>Loading MNIST Test 10k data...</div>"),In=Ue("<div> </div>"),Bn=Ue("<!> <div></div>",1);function Pn(t,e){Pt(e,!1);let r=pt(!1);async function n(){Me(r,!0);const p=await(await fetch("mnist_test.json")).json(),y=p.x,E=p.y;return Me(r,!1),[y,E]}function s(g,p){const P=K.tensor(_.randn([g,p],0,.1),!0),$=K.tensor(_.randn([1,p],0,.1),!0);return[P,$]}function i(g,p=[728,128],y=32,E=10){let P=[];for(let k=0;k<p.length-1;k++){const[U,he]=s(p[k],p[k+1]);P.push(U),P.push(he),g=g.matmul(U).add(he.expand([y,p[k+1]])).relu()}const[$,R]=s(p.at(-1),E);return P.push($),P.push(R),[g.matmul($).add(R.expand([y,E])).softmax(-1),P]}function a(g,p,y=32){return g.log().mul(p).sum(-1).sum(0).mul(-1/y)}function u(g){const p=new Float32Array(g.length*g[0].length);for(let y=0;y<g.length;y++)for(let E=0;E<g[0].length;E++)p[y*g[0].length+E]=g[y][E];return p}const o=64,f=.1,c=5;let d=pt([]);dn(async()=>{const[g,p]=await n(),y=K.tensor(_.fill(1,[o,28*28])),[E,P]=i(y,[28*28,128,128],o),$=K.tensor(_.fill(1,[o,10])),R=a(E,$,o),Z=new An(P,f),J=(k,U)=>u(k.slice(U,U+o));for(let k=0;k<c;k++){console.log("EPOCH"+(k+1));for(let U=0;U<g.length-o;U+=o)if(y.tensor.setGPUBuffer(J(g,U)),$.tensor.setGPUBuffer(J(p,U)),R.forward(),R.zeroGrad(),R.backward(),Z.update(),U%10===0){const he=await R.tensor.cpuBuffer();F(d).push(he[0]),Me(d,F(d))}}}),cn();var l=Bn(),h=Ir(l);{var m=g=>{var p=$n();ve(g,p)};Zt(h,g=>{F(r)&&g(m)})}var B=Br(h,2);un(B,5,()=>F(d),sn,(g,p)=>{var y=In(),E=Rt(y);Rr(()=>en(E,`Loss: ${F(p)??""}`)),ve(g,y)}),ve(t,l),St()}var Sn=Ue("<main><!></main>");function Dn(t){var e=Sn(),r=Rt(e);{var n=s=>{Pn(s,{})};Zt(r,s=>{s(n)})}ve(t,e)}tn(Dn,{target:document.getElementById("app")});
