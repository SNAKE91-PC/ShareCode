



function do_POST_ajax (entryPoint, query, callback)
{
    let req = new XMLHttpRequest();
    /*var result = document.getElementById(storeElement);*/
    req.onreadystatechange = function()
    {
        if(this.readyState == 4 && this.status == 200) {
            callback(this.responseText); /*result.innerHTML = */
        }
    }

    req.open('POST', entryPoint, true);
    req.setRequestHeader('content-type', 'application/x-www-form-urlencoded;charset=UTF-8');


    /* execute script received from server */
    /*https://stackoverflow.com/questions/3728798/running-javascript-downloaded-with-xmlhttprequest*/



    req.send(query);
};




function do_GET_ajax(entryPoint, query, postprocessing)
{
    var req = new XMLHttpRequest();
    var result = document.getElementById(storeElement);
    req.onreadystatechange = function()
    {
        if(this.readyState == 4 && this.status == 200) {
            postprocessing(this.responseText);
        }
    }

    req.open('GET', entryPoint + query, true);
    /*req.setRequestHeader('content-type', 'application/x-www-form-urlencoded;charset=UTF-8'); */


    /* execute script received from server */
    /*https://stackoverflow.com/questions/3728798/running-javascript-downloaded-with-xmlhttprequest*/

    req.send();
};





/*
document.addEventListener('DOMContentLoaded', function()
{
    document.getElementById("btn-post").addEventListener("click", function()
    {
        do_POST_ajax();

    })

    document.querySelector('form').addEventListener("submit", function(event)
    {
        event.preventDefault();
        do_POST_ajax();
    });

    document.getElementById("btn-query").addEventListener("click", function()
    {
        do_GET_ajax();

    })


})
*/