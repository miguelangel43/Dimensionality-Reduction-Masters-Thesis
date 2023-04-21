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

