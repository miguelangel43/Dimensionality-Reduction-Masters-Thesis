library(tidyverse)
library(gapminder)
library('dplyr')
theme_set(theme_bw(16))

# Group plots

df <- read.csv("/Users/espina/Downloads/corr_1dim.csv")

slmvp <- df %>% select("X1Dim.SLMVP.Polynomial.Order.5", "X")
slmvp <- rename(slmvp, Correlation = X1Dim.SLMVP.Polynomial.Order.5)
slmvp <- rename(slmvp, feature = X)
slmvp['dim_model'] = 'SLMVP'
slmvp <- tibble::rowid_to_column(slmvp, "paired")

lle <- df %>% select("X1Dim.LLE.k.68.reg.0.001", "X")
lle <- rename(lle, Correlation = X1Dim.LLE.k.68.reg.0.001)
lle <- rename(lle, feature = X)
lle['dim_model'] = 'LLE'
lle <- tibble::rowid_to_column(lle, "paired")

df_plot <- union(slmvp, lle)
df_plot <- read.csv("/Users/espina/Downloads/df_plot1.csv")

df_plot <- read.csv("/Users/espina/Desktop/R_CSVs/input_lol.csv")

#write.csv(df_plot, 'df_plot.csv')

df_plot %>%
  ggplot(aes(dim_model,inv_rank, fill=dim_model, label=feature, color=color)) +
  scale_fill_manual(breaks = c("0", "1", "2"), 
                    values=c("red", "green")) +
  geom_text(check_overlap = FALSE,
            size = 3,
            position=position_jitter(width=0.25))+
  geom_point()+ 
  geom_line(aes(group=paired, color=color)) +
  theme(legend.position = "none") 

# ------------------ Linear Regression Betas -------------------
library(dplyr)

betas <- read.csv("/Users/espina/Desktop/R_CSVs/linear_regression_betas.csv")
betas$X1 <- 0
plot(betas['Beta'], xlab="Betas ", pch=19, method="jitter", label=betas['Features'])

ggplot(data = betas) +
  geom_point(mapping = aes(x = Beta, y = X1),
             position = position_jitter(width = 5))
  
  

ggplot(data = betas, aes(x = Beta)) +
  geom_dotplot(binaxis = "x", stackdir="center", dotsize = 0.5,
               position = position_jitter(height=0.5)) 

  geom_text(x=betas$Beta, y=betas$X1 ,check_overlap = FALSE, label=betas$Features)

# ------------------ ChatGPT Plots -------------------

# Create a sample ranking data
rank_data <- c(5, 3, 1, 4, 2)

# Create a bar plot
barplot(rank_data, horiz = FALSE, xlab = "Ranking", ylab = "Value")
plot(rank_data, type = "p", pch = 16, xlab = "Value", ylab = "Ranking")  
# Sort the data in descending order
sorted_data <- sort(rank_data, decreasing = TRUE)

# Create a ranked bar plot
barplot(sorted_data, horiz = FALSE, xlab = "Value", ylab = "Ranking", names.arg = 1:length(sorted_data))  

# Create a sample ranking data for two items over time
item1 <- c(3, 2, 1, 4, 5)
item2 <- c(2, 4, 1, 3, 5)
time <- c(1, 2, 3, 4, 5)

# Create a rank chart
plot(time, item1, type = "b", pch = 16, xlab = "Time", ylab = "Ranking")
lines(time, item2, type = "b", pch = 16, col = "red")
legend("topright", legend = c("Item 1", "Item 2"), col = c("black", "red"), pch = 16)

# Create a sample ranking data for two items over time
item1 <- c(3, 2, 1, 4, 5)
item2 <- c(2, 4, 1, 3, 5)
time <- c(1, 2, 3, 4, 5)

# Create an area chart
plot(time, item1, type = "n", ylim = c(1, max(item1)), xlab = "Time", ylab = "Ranking")
for (i in 1:length(item1)) {
  polygon(c(time[i], time[i], time[i+1], time[i+1]), c(0, item1[i], item1[i], 0), col = "blue", border = NA)
  polygon(c(time[i], time[i], time[i+1], time[i+1]), c(0, item2[i], item2[i], 0), col = "red", border = NA)
}
legend("topright", legend = c("Item 1", "Item 2"), col = c("blue", "red"), fill = c("blue", "red"), border = NA)

# Create a sample ranking data for three items
item1 <- c(1, 2, 3, 4)
item2 <- c(2, 1, 4, 3)
item3 <- c(3, 4, 2, 1)

# Combine the data into a matrix
rank_matrix <- rbind(item1, item2, item3)

# Create a heatmap
heatmap(rank_matrix, xlab = "Time", ylab = "Item", col = heat.colors(max(rank_matrix)), scale = "none")

# Create data: note in High school for Jonathan:
data <- as.data.frame(matrix( sample( 2:20 , 10 , replace=T) , ncol=10))
colnames(data) <- c("math" , "english" , "biology" , "music" , "R-coding", "data-viz" , "french" , "physic", "statistic", "sport" )

# To use the fmsb package, I have to add 2 lines to the dataframe: the max and min of each topic to show on the plot!
data <- rbind(rep(20,10) , rep(0,10) , data)

# Check your data, it has to look like this!
# head(data)

# The default radar chart 
radarchart(data)

# ------------------  -------------------


