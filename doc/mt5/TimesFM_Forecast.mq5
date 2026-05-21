//+------------------------------------------------------------------+
//|                                          TimesFM_Forecast.mq5    |
//|                        TimesFM 2.5 Forecast Overlay Indicator    |
//|                                                                  |
//| Reads forecast CSVs from MQL5/Files/TimesFM/ and draws:          |
//|   - Point forecast line (solid, green=bull / red=bear)           |
//|   - 90% confidence band (shaded channel between q10 and q90)    |
//|   - Median line (q50, dotted)                                   |
//|   - Horizontal reference line at last close                     |
//|                                                                  |
//| Uses chart OBJECTS so forecasts render on future (unformed) bars.|
//|                                                                  |
//| Usage:                                                           |
//|   1. Run: python mt5/scripts/forecast.py                         |
//|   2. Run: python mt5/scripts/export_to_mt5.py                    |
//|   3. Attach this indicator to the chart                          |
//|   4. Press "R" to manually reload after a new forecast           |
//+------------------------------------------------------------------+
#property copyright   "TimesFM MT5 Project"
#property version     "2.00"
#property description "TimesFM 2.5 forecast overlay with confidence bands"
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

//--- inputs
input color    InpBullColor     = clrLime;           // Bull forecast color
input color    InpBearColor     = clrTomato;         // Bear forecast color
input color    InpBandColor     = clrDodgerBlue;     // Confidence band color
input color    InpMedianColor   = clrWhite;          // Median (q50) color
input int      InpForecastWidth = 3;                 // Forecast line width
input int      InpBandWidth     = 1;                 // Band border width
input bool     InpAutoReload    = true;              // Auto-reload on new bar
input string   InpSubfolder     = "TimesFM";         // Subfolder in MQL5/Files

//--- object name prefix (for cleanup)
#define OBJ_PREFIX "TFM_"

//--- state
string   g_symbol;
int      g_horizon;
datetime g_forecast_times[];
double   g_point[];
double   g_q10[];
double   g_q50[];
double   g_q90[];
datetime g_last_bar_time;
bool     g_loaded;
string   g_last_update;
double   g_last_close;

//+------------------------------------------------------------------+
int OnInit()
{
   g_symbol = Symbol();
   g_loaded = false;
   g_last_bar_time = 0;
   g_last_close = 0;

   // Load and draw
   if(LoadForecastCSV())
   {
      g_loaded = true;
      DrawForecast();
      Print("TimesFM: Loaded forecast for ", g_symbol, " (", g_horizon, " bars)");
   }
   else
   {
      Print("TimesFM: No forecast file found for ", g_symbol);
      Print("TimesFM: Expected: MQL5/Files/", InpSubfolder, "/", g_symbol, "_forecast.csv");
   }

   IndicatorSetString(INDICATOR_SHORTNAME, "TimesFM(" + g_symbol + ")");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
bool LoadForecastCSV()
{
   string filename = InpSubfolder + "\\" + g_symbol + "_forecast.csv";
   int handle = FileOpen(filename, FILE_READ | FILE_CSV | FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
      return false;

   // Skip header line (read all fields until line ends)
   while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
      FileReadString(handle);

   // Read data rows
   int count = 0;
   double point_arr[], q10_arr[], q50_arr[], q90_arr[];
   datetime dt_arr[];

   while(!FileIsEnding(handle))
   {
      string dt_str = FileReadString(handle);
      if(StringLen(dt_str) < 10) break;

      string pt_str  = FileReadString(handle);
      string q10_str = FileReadString(handle);
      string q50_str = FileReadString(handle);
      string q90_str = FileReadString(handle);

      datetime dt = StringToTime(dt_str);
      if(dt == 0) continue;

      count++;
      ArrayResize(dt_arr,    count);
      ArrayResize(point_arr, count);
      ArrayResize(q10_arr,   count);
      ArrayResize(q50_arr,   count);
      ArrayResize(q90_arr,   count);

      dt_arr[count-1]    = dt;
      point_arr[count-1] = StringToDouble(pt_str);
      q10_arr[count-1]   = StringToDouble(q10_str);
      q50_arr[count-1]   = StringToDouble(q50_str);
      q90_arr[count-1]   = StringToDouble(q90_str);
   }

   FileClose(handle);

   if(count == 0)
      return false;

   g_horizon = count;
   ArrayResize(g_forecast_times, count);
   ArrayResize(g_point, count);
   ArrayResize(g_q10,   count);
   ArrayResize(g_q50,   count);
   ArrayResize(g_q90,   count);

   ArrayCopy(g_forecast_times, dt_arr);
   ArrayCopy(g_point, point_arr);
   ArrayCopy(g_q10,   q10_arr);
   ArrayCopy(g_q50,   q50_arr);
   ArrayCopy(g_q90,   q90_arr);

   // Read last update time
   string meta_file = InpSubfolder + "\\last_update.txt";
   int mh = FileOpen(meta_file, FILE_READ | FILE_TXT | FILE_ANSI);
   if(mh != INVALID_HANDLE)
   {
      g_last_update = FileReadString(mh);
      FileClose(mh);
   }

   return true;
}

//+------------------------------------------------------------------+
void DeleteAllObjects()
{
   int total = ObjectsTotal(0, 0, -1);
   for(int i = total - 1; i >= 0; i--)
   {
      string name = ObjectName(0, i, 0, -1);
      if(StringFind(name, OBJ_PREFIX) == 0)
         ObjectDelete(0, name);
   }
}

//+------------------------------------------------------------------+
void CreateTrendLine(string name, datetime t1, double p1,
                     datetime t2, double p2,
                     color clr, int width, ENUM_LINE_STYLE style)
{
   ObjectCreate(0, name, OBJ_TREND, 0, t1, p1, t2, p2);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
   ObjectSetInteger(0, name, OBJPROP_STYLE, style);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}

//+------------------------------------------------------------------+
void CreateBandTriangle(string name, datetime t1, double p1,
                        datetime t2, double p2,
                        datetime t3, double p3,
                        color clr)
{
   ObjectCreate(0, name, OBJ_TRIANGLE, 0, t1, p1, t2, p2, t3, p3);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_FILL, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 0);
}

//+------------------------------------------------------------------+
void DrawForecast()
{
   if(g_horizon < 2) return;

   // Remove old objects
   DeleteAllObjects();

   // Determine direction
   double last_price = iClose(g_symbol, PERIOD_CURRENT, 0);
   if(last_price == 0) last_price = g_point[0];
   g_last_close = last_price;

   bool is_bull = (g_point[g_horizon-1] > last_price);
   color fc_color = is_bull ? InpBullColor : InpBearColor;

   // Get the time of the last bar to draw a connection line
   datetime last_bar_dt = iTime(g_symbol, PERIOD_CURRENT, 0);
   if(last_bar_dt > 0)
   {
      // Connection: last bar close -> first forecast point
      CreateTrendLine(OBJ_PREFIX + "conn",
                      last_bar_dt, last_price,
                      g_forecast_times[0], g_point[0],
                      fc_color, 2, STYLE_DOT);
   }

   // Draw confidence band as filled triangles between q10 and q90
   // Each pair of adjacent forecast points creates a quad, drawn as 2 triangles
   for(int i = 0; i < g_horizon - 1; i++)
   {
      datetime t1 = g_forecast_times[i];
      datetime t2 = g_forecast_times[i+1];

      // Triangle 1: (t1,q90[i]) - (t2,q90[i+1]) - (t1,q10[i])
      CreateBandTriangle(
         OBJ_PREFIX + "band1_" + IntegerToString(i),
         t1, g_q90[i],  t2, g_q90[i+1],  t1, g_q10[i],
         InpBandColor);

      // Triangle 2: (t2,q90[i+1]) - (t2,q10[i+1]) - (t1,q10[i])
      CreateBandTriangle(
         OBJ_PREFIX + "band2_" + IntegerToString(i),
         t2, g_q90[i+1],  t2, g_q10[i+1],  t1, g_q10[i],
         InpBandColor);
   }

   // Draw point forecast line segments
   for(int i = 0; i < g_horizon - 1; i++)
   {
      CreateTrendLine(
         OBJ_PREFIX + "fc_" + IntegerToString(i),
         g_forecast_times[i],   g_point[i],
         g_forecast_times[i+1], g_point[i+1],
         fc_color, InpForecastWidth, STYLE_SOLID);
   }

   // Draw median (q50) line segments
   for(int i = 0; i < g_horizon - 1; i++)
   {
      CreateTrendLine(
         OBJ_PREFIX + "med_" + IntegerToString(i),
         g_forecast_times[i],   g_q50[i],
         g_forecast_times[i+1], g_q50[i+1],
         InpMedianColor, 1, STYLE_DOT);
   }

   // Horizontal reference line at last close
   if(last_bar_dt > 0)
   {
      CreateTrendLine(
         OBJ_PREFIX + "ref",
         last_bar_dt, last_price,
         g_forecast_times[g_horizon-1], last_price,
         clrGray, 1, STYLE_DASH);
   }

   // Label with forecast info
   string dir_text = is_bull ? "LONG" : "SHORT";
   double pct = (g_point[g_horizon-1] / last_price - 1.0) * 100.0;
   string label_text = "TimesFM: " + dir_text + " "
                       + DoubleToString(MathAbs(pct), 2) + "% | "
                       + IntegerToString(g_horizon) + "h";
   if(StringLen(g_last_update) > 0)
      label_text += " | Updated: " + g_last_update;

   ObjectCreate(0, OBJ_PREFIX + "label", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, OBJ_PREFIX + "label", OBJPROP_XDISTANCE, 15);
   ObjectSetInteger(0, OBJ_PREFIX + "label", OBJPROP_YDISTANCE, 25);
   ObjectSetInteger(0, OBJ_PREFIX + "label", OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetString(0,  OBJ_PREFIX + "label", OBJPROP_TEXT, label_text);
   ObjectSetString(0,  OBJ_PREFIX + "label", OBJPROP_FONT, "Consolas");
   ObjectSetInteger(0, OBJ_PREFIX + "label", OBJPROP_FONTSIZE, 11);
   ObjectSetInteger(0, OBJ_PREFIX + "label", OBJPROP_COLOR, fc_color);
   ObjectSetInteger(0, OBJ_PREFIX + "label", OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, OBJ_PREFIX + "label", OBJPROP_HIDDEN, true);

   // End price label at the last forecast point
   string end_label = DoubleToString(g_point[g_horizon-1], (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS));
   ObjectCreate(0, OBJ_PREFIX + "endprice", OBJ_TEXT, 0,
                g_forecast_times[g_horizon-1], g_point[g_horizon-1]);
   ObjectSetString(0,  OBJ_PREFIX + "endprice", OBJPROP_TEXT, "  " + end_label);
   ObjectSetString(0,  OBJ_PREFIX + "endprice", OBJPROP_FONT, "Consolas");
   ObjectSetInteger(0, OBJ_PREFIX + "endprice", OBJPROP_FONTSIZE, 9);
   ObjectSetInteger(0, OBJ_PREFIX + "endprice", OBJPROP_COLOR, fc_color);
   ObjectSetInteger(0, OBJ_PREFIX + "endprice", OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, OBJ_PREFIX + "endprice", OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, OBJ_PREFIX + "endprice", OBJPROP_ANCHOR, ANCHOR_LEFT);

   ChartRedraw();
}

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   // Auto-reload on new bar
   if(InpAutoReload && rates_total > 0)
   {
      if(time[rates_total-1] != g_last_bar_time)
      {
         g_last_bar_time = time[rates_total-1];
         if(LoadForecastCSV())
         {
            g_loaded = true;
            DrawForecast();
            Print("TimesFM: Reloaded forecast on new bar");
         }
      }
   }

   return rates_total;
}

//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
   // Press 'R' to manually reload
   if(id == CHARTEVENT_KEYDOWN && lparam == 'R')
   {
      if(LoadForecastCSV())
      {
         g_loaded = true;
         DrawForecast();
         Print("TimesFM: Manual reload — ", g_horizon, " bars drawn");
      }
      else
         Print("TimesFM: Reload failed — no CSV for ", g_symbol);
   }
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Remove all our objects on indicator removal
   DeleteAllObjects();
   ChartRedraw();
   Comment("");
}
//+------------------------------------------------------------------+
