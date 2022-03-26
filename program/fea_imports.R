library(ggplot2)
databar1<-read.table('../results/each_cancer_fea.txt', header=T, sep='\t', encoding="UTF-8")
print(databar1)
theme_bar <- function(..., bg='white'){
  require(grid)
  theme_classic(...) +
    theme(rect=element_rect(fill=bg),
          plot.margin=unit(rep(0.5,4), 'lines'),
          panel.background=element_rect(fill='transparent', color='black'),
          plot.title = element_text(hjust = 0.5, face = "bold",size = 14),
          panel.border=element_rect(fill='transparent', color='transparent'),
          panel.grid=element_blank(),
          #axis.title.x = element_blank(),
          axis.title.y=element_text(face = "bold",size = 14),
          axis.title.x=element_text(face = "bold",size = 14),
          axis.title=element_text(face = "bold",size = 14),
          axis.text = element_text(face = "bold",size = 12),
          axis.text.x = element_text(angle=45, hjust = 0.5, vjust = 0.5, face = "bold",size = 12),
          # axis.ticks = element_line(color='black'),
          # axis.ticks.margin = unit(0.8,"lines"),
          legend.title=element_blank(),
          # legend.justification=c(1,0),
          legend.position=c(0.87, 0.9),
          # legend.position='top',
          legend.direction = "vertical",
          # legend.spacing.x = unit(2, 'cm'),
          legend.text = element_text(face = "bold",size = 12,margin = margin(r=20)),
          legend.background = element_rect( linetype="solid",colour ="black")
          # legend.margin=margin(0,0,-7,0)
          # legend.box.margin =margin(-10,0,0,0)
    )
  
}
q2<-ggplot(data=databar1, mapping=aes(x = Cancer, y = Importance,fill=Features))+
    geom_bar(stat="identity",position=position_dodge(0.75))+
    labs(title = "Features importance")+
    theme_bar()+ 
    coord_cartesian(ylim=c(0,0.75))+
    scale_y_continuous(expand = c(0, 0))
q2
#theme函数设置
