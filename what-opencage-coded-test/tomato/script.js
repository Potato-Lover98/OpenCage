let tomato = document.getElementById('tomato');
let clickCount = 0;

tomato.addEventListener('click', () => {
    clickCount++;
    console.log(`You clicked the tomato ${clickCount} times!`);
    tomato.style.background = `hsl(${clickCount * 10}, 100%, 50%)`;
});