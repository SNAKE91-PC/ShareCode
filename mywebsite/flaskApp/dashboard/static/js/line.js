

// set the dimensions and margins of the graph



function lineChart(queryString, typeQuery)
{
    
    let margin = {top: 30, right: 20, bottom: 70, left: 50},
    width = 600 - margin.left - margin.right,
    height = 300 - margin.top - margin.bottom;

    // Parse the date / time
    let parseDate = d3.timeParse("%b %Y");

    // Set the ranges
    let x = d3.scaleTime().range([0, width]);
    let y = d3.scaleLinear().range([height, 0]);

    // Define the line
    let priceline = d3.line()
    .x(function(d) { return x(d.Date); })
    .y(function(d) { return y(d.Close); });

    // Adds the svg canvas
    let svg = d3.select("body")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
    "translate(" + margin.left + "," + margin.top + ")");

    // Get the data

    let promiseData;

    if (typeQuery === "GET")
    {
        promiseData = d3.json("/query" + queryString)
    }
    if (typeQuery === "POST")
    {
        promiseData = d3.json("/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: [queryString]
            })

    }

    promiseData.then(function(data) {

    data.forEach(function(d) {
        parseDate = d3.utcParse("%Y-%m-%d")
        d.Date = parseDate(d.Date);
        d.Close = +d.Close;
    });

    // Scale the range of the data
    x.domain(d3.extent(data, function(d) { return d.Date; }));
    y.domain([0, d3.max(data, function(d) { return d.Close; })]);

    // Group the entries by symbol
    dataNest = Array.from(
    d3.group(data, d => d.symbol), ([key, value]) => ({key, value})
    );

    // set the colour scale
    let color = d3.scaleOrdinal(d3.schemeCategory10);

    legendSpace = width/dataNest.length; // spacing for the legend

    // Loop through each symbol / key
    dataNest.forEach(function(d,i) {

    svg.append("path")
    .attr("class", "line")
    .style("stroke", function()
        { // Add the colours dynamically
        return d.color = color(d.key);
        })
    .attr("id",
        'tag'+d.key.replace(/\s+/g, '')
        ) // assign an ID
    .attr("d", priceline(d.value));

    // Add the Legend
    svg.append("text")
    .attr("x", (legendSpace/2)+i*legendSpace)  // space legend
    .attr("y", height + (margin.bottom/2)+ 5)
    .attr("class", "legend")    // style the legend
    .style("fill", function() { // Add the colours dynamically
    return d.color = color(d.key); })
    .on("click", function(){
    // Determine if current line is visible
    let active   = d.active ? false : true,
    newOpacity = active ? 0 : 1;
    // Hide or show the elements based on the ID
    d3.select("#tag"+d.key.replace(/\s+/g, ''))
    .transition().duration(100)
    .style("opacity", newOpacity);
    // Update whether or not the elements are active
    d.active = active;
})
    .text(d.key);

});

    // Add the X Axis
    svg.append("g")
    .attr("class", "axis")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

    // Add the Y Axis
    svg.append("g")
    .attr("class", "axis")
    .call(d3.axisLeft(y));

});

    

}
