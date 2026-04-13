window.Module = { onRuntimeInitialized() { initApp(); } };

function initApp(){

const fileInput=document.getElementById("fileInput");
const originalCanvas=document.getElementById("originalCanvas");
const noisyCanvas=document.getElementById("noisyCanvas");
const restoredCanvas=document.getElementById("restoredCanvas");

const noiseType=document.getElementById("noiseType");
const noiseLevel=document.getElementById("noiseLevel");
const filterType=document.getElementById("filterType");

const psnrVal=document.getElementById("psnrVal");
const ssimVal=document.getElementById("ssimVal");

const statusText=document.getElementById("statusText");
const statusDot=document.getElementById("statusDot");

let originalGray=null;
let noisyGray=null;

function setReady(){ statusText.textContent="OpenCV.js Ready"; statusDot.style.background="#22c55e"; }
function setProcessing(){ statusText.textContent="Processing..."; statusDot.style.background="#ef4444"; }

function safeDisplay(mat, canvas){
  let rgba=new cv.Mat();
  cv.cvtColor(mat, rgba, cv.COLOR_GRAY2RGBA);
  cv.imshow(canvas, rgba);
  rgba.delete();
}

function gaussianRandom(mean=0, std=1){
  let u1=Math.random();
  let u2=Math.random();
  let z=Math.sqrt(-2*Math.log(u1))*Math.cos(2*Math.PI*u2);
  return z*std+mean;
}

fileInput.addEventListener("change",e=>{
const img=new Image();
img.onload=()=>{
originalCanvas.width=img.width;
originalCanvas.height=img.height;
noisyCanvas.width=img.width;
noisyCanvas.height=img.height;
restoredCanvas.width=img.width;
restoredCanvas.height=img.height;

originalCanvas.getContext("2d").drawImage(img,0,0);
let temp=cv.imread(originalCanvas);
originalGray=new cv.Mat();
cv.cvtColor(temp,originalGray,cv.COLOR_RGBA2GRAY);
safeDisplay(originalGray, originalCanvas);
temp.delete();
setReady();
};
img.src=URL.createObjectURL(e.target.files[0]);
});

document.getElementById("addNoiseBtn").onclick=()=>{

if(!originalGray) return;
setProcessing();

noisyGray=new cv.Mat();
originalGray.copyTo(noisyGray);

if(noiseType.value==="gaussian"){
let std=parseInt(noiseLevel.value);
for(let i=0;i<noisyGray.rows;i++){
  for(let j=0;j<noisyGray.cols;j++){
    let val=noisyGray.ucharPtr(i,j)[0]+gaussianRandom(0,std);
    noisyGray.ucharPtr(i,j)[0]=Math.max(0,Math.min(255,val));
  }
}
}

if(noiseType.value==="sp"){
let prob=noiseLevel.value/100;
for(let i=0;i<noisyGray.rows;i++){
for(let j=0;j<noisyGray.cols;j++){
if(Math.random()<prob){
noisyGray.ucharPtr(i,j)[0]=Math.random()<0.5?0:255;
}
}
}
}

safeDisplay(noisyGray, noisyCanvas);
setReady();
};

document.getElementById("applyFilterBtn").onclick=()=>{

if(!noisyGray) return;
setProcessing();

let restored=new cv.Mat();

if(filterType.value==="gaussian") cv.GaussianBlur(noisyGray,restored,new cv.Size(5,5),0);
if(filterType.value==="median") cv.medianBlur(noisyGray,restored,5);
if(filterType.value==="bilateral") cv.bilateralFilter(noisyGray,restored,9,75,75);
if(filterType.value==="nlm") cv.fastNlMeansDenoising(noisyGray,restored,10,7,21);

safeDisplay(restored, restoredCanvas);
computeMetrics(originalGray,restored);

restored.delete();
setReady();
};

function computeMetrics(orig,rest){

let diff=new cv.Mat();
cv.absdiff(orig,rest,diff);
diff.convertTo(diff,cv.CV_32F);
cv.multiply(diff,diff,diff);

let mse=cv.mean(diff)[0];
mse/=255*255;

let psnr=10*Math.log10(1/mse);
psnrVal.textContent=psnr.toFixed(2);
ssimVal.textContent=computeSSIM(orig,rest).toFixed(3);

diff.delete();
}

function computeSSIM(img1,img2){

let mean1=cv.mean(img1)[0];
let mean2=cv.mean(img2)[0];
let var1=0,var2=0,cov=0;
let N=img1.rows*img1.cols;

for(let i=0;i<img1.rows;i++){
for(let j=0;j<img1.cols;j++){
let p1=img1.ucharPtr(i,j)[0];
let p2=img2.ucharPtr(i,j)[0];
var1+=(p1-mean1)*(p1-mean1);
var2+=(p2-mean2)*(p2-mean2);
cov+=(p1-mean1)*(p2-mean2);
}
}

var1/=N; var2/=N; cov/=N;
let C1=6.5025,C2=58.5225;
return ((2*mean1*mean2+C1)*(2*cov+C2))/((mean1*mean1+mean2*mean2+C1)*(var1+var2+C2));
}

document.getElementById("resetBtn").onclick=()=>location.reload();
document.getElementById("downloadBtn").onclick=()=>{
const link=document.createElement("a");
link.download="restored.png";
link.href=restoredCanvas.toDataURL();
link.click();
};
document.getElementById("fullscreenBtn").onclick=()=>document.documentElement.requestFullscreen();

}
