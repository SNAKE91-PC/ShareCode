#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(reticulate)

# Define UI for application that draws a histogram
ui <- fluidPage(
    
    # Application title
    titlePanel("KS"),
    
    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
            sliderInput("bins", "Number of bins:", min = 1, max = 500, value = 100),
            sliderInput("nseries", "Number of series", min = 1, max = 10, value = 5),
            sliderInput("nobs", "Number of obs", min = 10, max = 50, value = 100),
            sliderInput("nsimulation", "Number of simulations", min = 2000, max = 10000, value = 5000)
        ),

        # Show a plot of the generated distribution
        mainPanel(
           plotOutput("distPlot")
        )
    )
)

rng <- runif(10000 * 100 * 10)
    
# Define server logic required to draw a histogram
server <- function(input, output) {

    output$distPlot <- renderPlot({
        # generate bins based on input$bins from ui.R
        #x    <- faithful[, 2]
        
        os = import("os")
        
        print(paste(os$getcwd(), "/stats/gof/ks", sep = ""))
        maxmv = import_from_path("maxmv", paste(os$getcwd(), "/stats/gof/ks", sep = ""))
        
        
        nseries = input$nseries
        nobs = input$nobs
        nsimulation = input$nsimulation
        x = array(dim = c(nsimulation, nobs))
        
        statsList = array(dim = c(nseries))
        rng <- rng[1:nsimulation] # assert or derive from nseries * nobs
        for(i in seq(1, nsimulation))
        {
            if(i==1)
            {
                x[i,] = rng[seq(nobs * (i-1), nobs * i)]
            }
            else
            {
                x[i,] = rng[seq(nobs * (i-1), nobs * i - 1)]
            }
        }
        
        c = 1
        for(j in seq(nseries, length(x[,1]) - input$nseries))
        {
            sample_theor_statsks = maxmv$nsample_theor_ks(x[(j+1-input$nseries): j,])
            statsList[c] = sample_theor_statsks #.append(sample_theor_statsks)
            
            c = c+1
            
            print(c)
        }
        
        bins <- seq(min(x), max(x), length.out = input$bins + 1)

        # draw the histogram with the specified number of bins
        hist(statsList, breaks = bins, col = 'darkgray', border = 'white')
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
