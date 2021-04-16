library(readxl)
geocoded_addresses <- read_excel("C:/Users/raven/OneDrive - FEMA/Desktop/SALAE Mapping/geocoded_addresses.xlsx")

library(leaflet)
#setting up quintile breaks
mybreaks <- c(q1 - min, q1, q3, q4, q5, q6 - max)
mypalette <- colorBin(palette = "YlOrRd", domain = geocoded_addresses$approved_to_date, na.color ="transparent", bins = mybreaks)

mytext <- paste(
  "Policy number: ", geocoded_addresses$policy_number, "<br>",
  "Date of Loss: ", geocoded_addresses$date_of_loss, "<br>",
  "Approval Expenses to Date: $", geocoded_addresses$approved_to_date) %>%
  lapply(htmltools::HTML)

m <- leaflet(geocoded_addresses) %>%
  addTiles() %>%
  setView(-95.483330, 35.712046, zoom = 4) %>%
  addCircleMarkers(~Longitude, ~Latitude,
                   fillColor = ~mypalette(approved_to_date), fillOpacity = 0.7, color = "White", radius = 4, stroke = FALSE,
                   label = mytext,) %>%
  addLegend( pal=mypalette, values=~approved_to_date, opacity = 0.9, title = "Expenses to Date (in USD)", position = "bottomright")
m