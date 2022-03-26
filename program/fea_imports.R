library(ggplot2)
databar1<-read.table('C:/Users/zhanglei/Desktop/论文材料/results_3/each_cancer_fea_3_4_11.txt', header=T, sep='\t', encoding="UTF-8")
print(databar1)
theme_bar <- function(..., bg='white'){
  require(grid)
  theme_classic(...) +
    theme(rect=element_rect(fill=bg),
          plot.margin=unit(rep(0.5,4), 'lines'),
          panel.background=element_rect(fill='transparent', color='black'),
          plot.title = element_text(hjust = 0.5, face = "bold",size = 14),
          panel.border=element_rect(fill='transparent', color='transparent'),
          panel.grid=element_blank(),#去网格线
          #axis.title.x = element_blank(),#去x轴标签
          axis.title.y=element_text(face = "bold",size = 14),#y轴标签加粗及字体大小
          axis.title.x=element_text(face = "bold",size = 14),#y轴标签加粗及字体大小
          axis.title=element_text(face = "bold",size = 14),#y轴标签加粗及字体大小
          axis.text = element_text(face = "bold",size = 12),#坐标轴刻度标签加粗
          axis.text.x = element_text(angle=45, hjust = 0.5, vjust = 0.5, face = "bold",size = 12),
          # axis.ticks = element_line(color='black'),#坐标轴刻度线
          # axis.ticks.margin = unit(0.8,"lines"),
          legend.title=element_blank(),#去除图例标题
          # legend.justification=c(1,0),#图例在画布的位置(绘图区域外)
          legend.position=c(0.87, 0.9),#图例在绘图区域的位置
          # legend.position='top',#图例放在顶部
          legend.direction = "vertical",#设置图例水平放置
          # legend.spacing.x = unit(2, 'cm'),
          legend.text = element_text(face = "bold",size = 12,margin = margin(r=20)),
          legend.background = element_rect( linetype="solid",colour ="black")
          # legend.margin=margin(0,0,-7,0)#图例与绘图区域边缘的距离
          # legend.box.margin =margin(-10,0,0,0)
    )
  
}
q2<-ggplot(data=databar1, mapping=aes(x = Cancer, y = Importance,fill=Features))+
    geom_bar(stat="identity",position=position_dodge(0.75))+
    labs(title = "Features importance")+
    theme_bar()+ 
    coord_cartesian(ylim=c(0,0.75))+
    scale_y_continuous(expand = c(0, 0))#消除x轴与绘图区的间隙
q2
#theme函数设置