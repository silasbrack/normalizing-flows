rm(list = ls())
library(tidyverse)
library(data.table)
library(ggthemes)
library(ggrepel)
library(RColorBrewer)
library(scales)
library(tikzDevice)
# dev.off()
library(extrafont)
loadfonts(device = "pdf")

# theme_set(theme_bw())
# theme_set(theme_minimal())
theme_set(
  # theme_minimal() +
  theme_bw() +
    theme(panel.grid = element_blank()) +
    theme(text = element_text(family = 'Helvetica'))
  )

elbo <- c(-33.406, -32.598, -33.297, -32.728, -32.412, -31.791, -37.756, -37.683, -35.904, -33.292)
khat <- c(0.8710, 0.8226, 0.8329, 0.8066, 0.7700, 0.6379, 0.9209, 0.8979, 0.8760, 0.8203)
num_flows <- c(NaN, NaN, 4, 8, 16, 32, 4, 8, 16, 32)
type <- c("Mean-field", "Full-rank", "Planar", "Planar", "Planar", "Planar", "Radial", "Radial", "Radial", "Radial")

dt <- data.table(data.frame(elbo, khat, num_flows, type))
dt$type = as.factor(dt$type)

dt_flow = dt %>% filter(type %in% c("Planar", "Radial"))
dt_rest = dt %>% filter(type %in% c("Mean-field", "Full-rank"))

# tikz(file = "./8schools_khat_gg.tex", width = 3, height = 3)
ggplot(dt_flow, aes(x=num_flows, y=khat, color=type)) +
  geom_point() +
  geom_line(linetype="dashed") +
  geom_text_repel(aes(label=type), data=dt%>%filter(num_flows == 8), show.legend=F, point.padding=1, box.padding=2) +
  scale_x_continuous(trans = "log2", labels = math_format(2^.x, format = log2)) +
  geom_hline(data=dt_rest, aes(yintercept=khat), linetype="dashed", color="grey") +
  geom_label(data=dt_rest, aes(label=type), x=c(2.3, 4), label.size=NA, color="grey") +
  geom_hline(yintercept=0.7, linetype="dashed", color="red") +
  labs(
    x="Number of flows",
    y="$\\hat{k}$",
  ) +
  theme(legend.position="none")
# dev.off()

# tikz(file = "./8schools_elbo_gg.tex", width = 3, height = 3)
ggplot(dt_flow, aes(x=num_flows, y=elbo, color=type)) +
  geom_point() +
  geom_line(linetype="dashed") +
  geom_label_repel(aes(label=type), data=dt%>%filter(num_flows == 8), label.size=F, show.legend=F, point.padding=1, box.padding=1) +
  geom_hline(data=dt_rest, aes(yintercept=elbo), linetype="dashed", color="grey") +
  geom_label(data=dt_rest, aes(label=type), x=c(3.7,2.3), label.size=NA, color="grey") +
  scale_x_continuous(trans = "log2", labels = math_format(2^.x, format = log2)) +
  labs(
    x="Number of flows",
    y="ELBO"
  ) +
  theme(legend.position="none")
# dev.off()

