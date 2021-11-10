library(data.table)
library(ggplot2)
library(RColorBrewer)
library(cowplot)
library(dplyr)
library(tikzDevice)

dt.loss <- data.table(read.csv("C:\\Users\\s174433\\Documents\\normalizing-flows\\results\\energy\\losses.csv"))
dt.int <- data.table(read.csv("C:\\Users\\s174433\\Documents\\normalizing-flows\\results\\energy\\test.csv"))

p1 <- ggplot(dt.loss, aes(x=X, y=elbo)) +
  geom_line() + 
  xlim(5000, 5500) + ylim(-50, 350) +
  theme_bw() +
  labs(x="Iterations", y="ELBO") +
  theme(
    panel.grid=element_blank(),
    legend.pos="none"
  )

p2 <- ggplot(dt.int %>% filter(iters==5000), aes(x=x, y=y, color=log_prob)) +
  geom_point(color="black", size=2) +
  geom_point() +
  xlim(-4, 4) + ylim(-4, 4) +
  scale_color_distiller(palette="YlOrRd") +
  labs(
    x="",
    y=""
  ) +
  theme_bw() +
  theme(
    panel.grid=element_blank(),
    legend.pos="none"
        )

p3 <- ggplot(dt.int %>% filter(iters==6000), aes(x=x, y=y, color=log_prob)) +
  geom_point(color="black", size=2) +
  geom_point() +
  xlim(-4, 4) + ylim(-4, 4) +
  scale_color_distiller(palette="YlOrRd") +
  labs(
    x="",
    y=""  ) +
  theme_bw() +
  theme(
    panel.grid=element_blank(),
    legend.pos="none"
  )

tikz(file="test.tex",width = 9, height = 3)
plot_grid(p1, p2, p3, labels="AUTO", ncol=3)
dev.off()