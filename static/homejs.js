window.addEventListener("load", function(){
    cardClick = document.querySelectorAll('.card')
    selectionLab = document.getElementById('selectionlab')
    selectionInp = document.getElementById('selectioninp')
      for (let i = 0; i < cardClick.length; i++) {
        cardClick[i].addEventListener("click", function() {
          var value = this.id.split("_")[0];
          selectionLab.innerHTML=value;
          selectionInp.value=value;
        });
      }
});