window.Module = {

    onRuntimeInitialized() {

        document.getElementById("statusText").innerText = "Ready";

        document.getElementById("statusDot").style.background = "#22c55e";

        console.log("OpenCV Ready");

        initApp();
    }

};

// window.Module = {
// onRuntimeInitialized() {
// document.getElementById("statusText").textContent="Ready";
// document.getElementById("statusDot").style.background="#22c55e";
// initApp();
// }
// };

function initApp(){

const originalCanvas=document.getElementById("original");
const blurredCanvas=document.getElementById("blurred");
const inverseCanvas=document.getElementById("inverse");
const wienerCanvas=document.getElementById("wiener");

const fileInput=document.getElementById("fileInput");
const psfType=document.getElementById("psfType");
const psfSize=document.getElementById("psfSize");
const wienerK=document.getElementById("wienerK");

let imgGray=null;

function setProcessing(){
document.getElementById("statusText").textContent="Processing...";
document.getElementById("statusDot").style.background="#ef4444";
}
function setReady(){
document.getElementById("statusText").textContent="Ready";
document.getElementById("statusDot").style.background="#22c55e";
}

function displayGray(mat, canvas){
let rgba=new cv.Mat();
cv.cvtColor(mat,rgba,cv.COLOR_GRAY2RGBA);
cv.imshow(canvas,rgba);
rgba.delete();
}

fileInput.addEventListener("change", e=>{
let img=new Image();
img.onload=()=>{
let maxDim=512;
let scale=Math.min(maxDim/img.width,maxDim/img.height,1);
let w=Math.floor(img.width*scale);
let h=Math.floor(img.height*scale);

[originalCanvas,blurredCanvas,inverseCanvas,wienerCanvas].forEach(c=>{c.width=w;c.height=h;});

let ctx=originalCanvas.getContext("2d");
ctx.drawImage(img,0,0,w,h);

let temp=cv.imread(originalCanvas);
imgGray=new cv.Mat();
cv.cvtColor(temp,imgGray,cv.COLOR_RGBA2GRAY);
displayGray(imgGray,originalCanvas);
temp.delete();
};
img.src=URL.createObjectURL(e.target.files[0]);
});

document.getElementById("processBtn").onclick=()=>{

if(!imgGray) return;
setProcessing();

let blurred=new cv.Mat();
let ksize=parseInt(psfSize.value);

if(psfType.value==="motion"){
let kernel=new cv.Mat.zeros(ksize,ksize,cv.CV_32F);
for(let i=0;i<ksize;i++){
kernel.floatPtr(Math.floor(ksize/2),i)[0]=1/ksize;
}
cv.filter2D(imgGray,blurred,cv.CV_8U,kernel);
kernel.delete();
}else{
cv.GaussianBlur(imgGray,blurred,new cv.Size(ksize,ksize),0);
}

displayGray(blurred,blurredCanvas);

// Inverse (simple sharpening approximation)
let inverse=new cv.Mat();
cv.addWeighted(imgGray,1.5,blurred,-0.5,0,inverse);
displayGray(inverse,inverseCanvas);

// Wiener (controlled attenuation)
let wiener=new cv.Mat();
let alpha=parseFloat(wienerK.value)*10;
cv.addWeighted(imgGray,1+alpha,blurred,-alpha,0,wiener);
displayGray(wiener,wienerCanvas);

blurred.delete();
inverse.delete();
wiener.delete();

setReady();
};
}
