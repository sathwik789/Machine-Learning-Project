
install.packages("shiny")
library(shiny)
library(caret)
library(class)
library(e1071)
library(ggplot2)

# Load data
weather_data <- read.csv(file.choose())

# Preprocess data
weather_data <- weather_data[, c("Precip.Type", "Temperature..C.", "Humidity", "Wind.Speed..km.h.")]
weather_data$Precip.Type[weather_data$Precip.Type == "null"] <- "Sunny"
weather_data <- weather_data[!duplicated(weather_data), ]
weather_data <- na.omit(weather_data)

# Normalize data
weather_data$Temperature..C. <- scale(weather_data$Temperature..C.)
weather_data$Humidity <- scale(weather_data$Humidity)
weather_data$Wind.Speed..km.h. <- scale(weather_data$Wind.Speed..km.h.)
weather_data$Precip.Type <- as.factor(weather_data$Precip.Type)

# Partition data
set.seed(123)
index <- createDataPartition(weather_data$Precip.Type, p = 0.8, list = FALSE)
training_set <- weather_data[index, ]
testing_set <- weather_data[-index, ]
training_labels <- weather_data[index, "Precip.Type"]
testing_labels <- weather_data[-index, "Precip.Type"]

# Define UI
ui <- fluidPage(
  titlePanel("Weather Prediction Dashboard"),
  sidebarLayout(
    sidebarPanel(
      h4("Model Settings"),
      numericInput("knn_k", "KNN: Number of Neighbors (k)", value = 3, min = 1, max = 10),
      actionButton("run_models", "Run Models"),
      br(),
      h4("Model Accuracy"),
      verbatimTextOutput("model_accuracy")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Data Overview", tableOutput("data_view")),
        tabPanel("Accuracy Comparison", plotOutput("accuracy_plot")),
        tabPanel("Confusion Matrices", 
                 h4("KNN Confusion Matrix"), plotOutput("knn_plot"),
                 h4("Naive Bayes Confusion Matrix"), plotOutput("nb_plot"))
      )
    )
  )
)

# Define server
server <- function(input, output) {
  # Reactive expression to run models
  models <- eventReactive(input$run_models, {
    # KNN model
    knn_model <- knn(training_set[-1], testing_set[-1], cl = training_labels, k = input$knn_k)
    knn_confus <- confusionMatrix(knn_model, testing_labels)
    knn_accuracy <- knn_confus$overall["Accuracy"]
    
    # Naive Bayes model
    naive_bayes <- naiveBayes(Precip.Type ~ ., data = training_set)
    nb_prediction <- predict(naive_bayes, testing_set)
    nb_confus <- confusionMatrix(nb_prediction, testing_labels)
    nb_accuracy <- nb_confus$overall["Accuracy"]
    
    # Return results
    list(knn_model = knn_model, nb_prediction = nb_prediction, knn_confus = knn_confus, 
         nb_confus = nb_confus, knn_accuracy = knn_accuracy, nb_accuracy = nb_accuracy)
  })
  
  # Display data
  output$data_view <- renderTable({
    head(weather_data)
  })
  
  # Display model accuracy
  output$model_accuracy <- renderPrint({
    results <- models()
    cat("KNN Accuracy:", round(results$knn_accuracy * 100, 2), "%\n")
    cat("Naive Bayes Accuracy:", round(results$nb_accuracy * 100, 2), "%\n")
  })
  
  # Accuracy comparison plot
  output$accuracy_plot <- renderPlot({
    results <- models()
    accuracy_df <- data.frame(
      Model = c("KNN", "Naive Bayes"),
      Accuracy = c(results$knn_accuracy * 100, results$nb_accuracy * 100)
    )
    ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
      geom_bar(stat = "identity", width = 0.6) +
      ylim(0, 100) +
      labs(title = "Model Accuracy Comparison", y = "Accuracy (%)", x = "Model") +
      theme_minimal() +
      scale_fill_manual(values = c("KNN" = "lightblue", "Naive Bayes" = "orange")) +
      geom_text(aes(label = round(Accuracy, 2)), vjust = -0.5)
  })
  
  # KNN confusion matrix plot
  output$knn_plot <- renderPlot({
    results <- models()
    knn_table <- as.data.frame(table(Predicted = results$knn_model, Actual = testing_labels))
    ggplot(knn_table, aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Freq), color = "black") +
      scale_fill_gradient(low = "white", high = "steelblue") +
      labs(title = "KNN Confusion Matrix", x = "Actual", y = "Predicted")
  })
  
  # Naive Bayes confusion matrix plot
  output$nb_plot <- renderPlot({
    results <- models()
    nb_table <- as.data.frame(table(Predicted = results$nb_prediction, Actual = testing_labels))
    ggplot(nb_table, aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Freq), color = "black") +
      scale_fill_gradient(low = "white", high = "tomato") +
      labs(title = "Naive Bayes Confusion Matrix", x = "Actual", y = "Predicted")
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
