window.addEventListener("load", function(){
    cardClick = document.querySelectorAll('.card')
    selectionBox = document.getElementById('selection')
      for (let i = 0; i < cardClick.length; i++) {
        cardClick[i].addEventListener("click", function() {
          var value = this.id.split("_")[0];
          selectionBox.value=value;
        });
      }
});