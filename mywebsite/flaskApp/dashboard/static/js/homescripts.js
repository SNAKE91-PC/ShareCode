


function add_btnplus()
{
    // First create a DIV element.
    /*var elem = document.getElementsByClassName("btn-plus")*/

    $('.btn-plus').remove();

    var txtNewInputBox = document.createElement('div');
    // Then add the content (a new input box) of the element.
    txtNewInputBox.innerHTML = "<label>Name:<input type=\"text\" class=\"input-name\" value=\"\" /></label> <button type=\"button\" class= \"btn-minus\">-</button> <button type=\"button\" class= \"btn-plus\">+</button>";


    // Finally put it where it is supposed to appear.
    document.getElementById("newElementId").appendChild(txtNewInputBox);
}


function onsubmit()
{
    const tickers = document.getElementsByClassName("input-name")

    /*RETRIEVING THE JSON FROM THE BOXES */
    console.log(tickers)
    let listTickers = []
    let queryString = ""

    for (let i = 0; i < tickers.length; i++)
    {
        listTickers.push(tickers[i].value)
        queryString += tickers[i].value

        if (tickers.length > 1)
        {
            queryString += "&"
        }
    }

    let typeQuery = "GET"

    if (typeQuery === "GET")
    {
        queryString = "?"+queryString
    }

    /*BUILDING POST REQUEST*/
    /*do_GET_ajax("table", "/query", queryString)*/

    lineChart(queryString, typeQuery)

    /*do_POST_ajax("/query", queryString, lineChart)*/


}



/*                      */



document.addEventListener('DOMContentLoaded', function() {


    const container = document.getElementById('form-button');

    container.addEventListener('click', function (e) {
        // But only alert for elements that have an alert-button class
        if (e.target.classList.contains('btn-plus'))
        {
            add_btnplus();
        }
        if (e.target.classList.contains('btn-minus'))
        {
            e.target.parentNode.remove()
            if (this.getElementsByClassName("btn-plus").length === 0)
            {

                /*console.log(this.lastElementChild)*/

                const btn_plus = document.createElement("btn-plus")
                btn_plus.innerHTML = "<button type=\"button\" class= \"btn-plus\">+</button> "

                const check = document.getElementById("newElementId")

                if (check > 1)
                {
                    let lastChild = check.lastChild
                    lastChild.insertAdjacentElement("beforeend", btn_plus)
                }
                if (check.children.length < 1)
                {
                    add_btnplus()
                    /*alert("Cannot have less than 1 stock in the portfolio")*/
                }




            }

        }

    });


    window.addEventListener('keypress', function (e){

        if (e.key === "+")
        {
            e.preventDefault()
            add_btnplus()
        }
        if (e.key === "-")
        {
            e.preventDefault()
            document.getElementById("newElementId").lastChild.remove();

            let check = document.getElementById("newElementId")

            if (check.children.length > 1)
            {
                const lastChild = check.lastChild

                let btn_plus = document.createElement("btn-plus")
                btn_plus.innerHTML = "<button type=\"button\" class= \"btn-plus\">+</button> "
                lastChild.insertAdjacentElement("beforeend", btn_plus)

            }
            if (check.children.length < 1)
            {
                add_btnplus()
                /*alert("Cannot have less than 1 stock in the portfolio")*/

            }

        }

    })


    document.getElementById("btn-go").addEventListener("click", function ()
    {
        onsubmit();

    })


    document.querySelector('form').addEventListener("submit", function(e)
    {
        e.preventDefault();
        onsubmit();
    });




})