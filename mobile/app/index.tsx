import {
  StyleSheet,
  View,
  Text,
  TextInput,
  ScrollView,
  Platform,
  Pressable,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { useState, useRef, useEffect } from "react";

// Conditionally import speech recognition (requires native build)
let ExpoSpeechRecognitionModule: any = null;
let useSpeechRecognitionEvent: any = null;

try {
  const speechRecognition = require("expo-speech-recognition");
  ExpoSpeechRecognitionModule = speechRecognition.ExpoSpeechRecognitionModule;
  useSpeechRecognitionEvent = speechRecognition.useSpeechRecognitionEvent;
} catch (error) {
  console.warn(
    "Speech recognition module not available. A development build is required."
  );
}

export default function HomeScreen() {
  const calendarScrollRef = useRef<ScrollView>(null);
  const [dates, setDates] = useState<Date[]>([]);
  const [transcript, setTranscript] = useState<string>("");
  const [isListening, setIsListening] = useState<boolean>(false);
  const [speechRecognitionAvailable, setSpeechRecognitionAvailable] =
    useState<boolean>(false);

  // Get today's date normalized to start of day
  const getToday = () => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return today;
  };

  // Generate dates: 30 days back from today (including today)
  useEffect(() => {
    const today = getToday();
    const dateArray: Date[] = [];
    for (let i = -30; i <= 0; i++) {
      const date = new Date(today);
      date.setDate(today.getDate() + i);
      dateArray.push(date);
    }
    setDates(dateArray);

    // Scroll to today's date on mount (today is at the last index)
    setTimeout(() => {
      const todayIndex = dateArray.length - 1; // Today is at the last index
      const itemWidth = 32; // width
      const itemGap = 12; // gap between items
      const scrollPosition = todayIndex * (itemWidth + itemGap) - 20; // Subtract padding
      calendarScrollRef.current?.scrollTo({
        x: Math.max(0, scrollPosition),
        animated: false,
      });
    }, 100);
  }, []);

  const formatDay = (date: Date) => {
    return date.getDate().toString();
  };

  const isToday = (date: Date) => {
    const today = getToday();
    return (
      date.getDate() === today.getDate() &&
      date.getMonth() === today.getMonth() &&
      date.getFullYear() === today.getFullYear()
    );
  };

  // Check if speech recognition is available
  useEffect(() => {
    setSpeechRecognitionAvailable(
      ExpoSpeechRecognitionModule !== null && useSpeechRecognitionEvent !== null
    );
  }, []);

  // Handle speech recognition results (only if available)
  useEffect(() => {
    if (!useSpeechRecognitionEvent) return;

    const handleResult = (event: any) => {
      const { transcript: newTranscript } = event;
      if (newTranscript) {
        setTranscript(newTranscript);
      }
    };

    const handleEnd = () => {
      setIsListening(false);
    };

    const handleError = (event: any) => {
      console.error("Speech recognition error:", event.error);
      setIsListening(false);
    };

    // Register event listeners
    useSpeechRecognitionEvent("result", handleResult);
    useSpeechRecognitionEvent("end", handleEnd);
    useSpeechRecognitionEvent("error", handleError);
  }, []);

  const startSpeechRecognition = async () => {
    if (!ExpoSpeechRecognitionModule) {
      alert(
        "Speech recognition requires a development build. Please build the app with 'npx expo prebuild' or use EAS Build."
      );
      return;
    }

    try {
      setIsListening(true);
      setTranscript("");
      await ExpoSpeechRecognitionModule.startAsync();
    } catch (error) {
      console.error("Error starting speech recognition:", error);
      setIsListening(false);
      alert("Failed to start speech recognition. Please check permissions.");
    }
  };

  return (
    <SafeAreaView style={styles.container} edges={["top", "bottom"]}>
      {/* White Content Area */}
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.whiteContent}>
          {/* Calendar Scroll */}
          <ScrollView
            ref={calendarScrollRef}
            horizontal
            showsHorizontalScrollIndicator={false}
            style={styles.calendarScrollView}
            contentContainerStyle={styles.calendarContainer}
            snapToInterval={44} // 32px width + 12px gap
            decelerationRate="fast"
            snapToAlignment="start"
          >
            {dates.map((date, index) => {
              const isTodayDate = isToday(date);
              const isLast = index === dates.length - 1;
              return (
                <View
                  key={index}
                  style={[
                    styles.tabNumber,
                    isTodayDate && styles.tabNumberToday,
                    isLast && styles.tabNumberLast,
                  ]}
                >
                  <Text
                    style={[
                      styles.tabNumberText,
                      isTodayDate && styles.tabNumberTextToday,
                    ]}
                  >
                    {formatDay(date)}
                  </Text>
                </View>
              );
            })}
          </ScrollView>

          {/* Speech Recognition Transcript */}
          {transcript ? (
            <View style={styles.transcriptContainer}>
              <Text style={styles.transcriptLabel}>You said:</Text>
              <View style={styles.transcriptCard}>
                <Text style={styles.transcriptText}>{transcript}</Text>
              </View>
            </View>
          ) : null}

          {/* Idea Entries */}
          <View style={styles.ideasContainer}>
            {[1, 2, 3].map((index) => (
              <View key={index} style={styles.ideaCard}>
                <Text style={styles.ideaText}>
                  Idea to build a speech-based AI notes app called Mindflow
                </Text>
              </View>
            ))}
          </View>

          {/* To-do Section */}
          <View style={{}}>
            <Text style={styles.todoHeading}>To-do:</Text>
            {[1, 2, 3, 4].map((index) => (
              <View key={index} style={styles.todoItem}>
                <View style={styles.checkbox} />
                <Text style={styles.todoText}>Build Mindflow</Text>
              </View>
            ))}
          </View>
        </View>
      </ScrollView>

      {/* Bottom Purple/Red Card */}
      <LinearGradient
        colors={["#8B5CF6", "#EC4899", "#EF4444"]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
        style={styles.blackCard}
      >
        <Pressable
          style={({ pressed }) => [
            styles.ctaButton,
            pressed && styles.ctaButtonPressed,
          ]}
          onPress={startSpeechRecognition}
          disabled={isListening}
        >
          <LinearGradient
            colors={["#FFFFFF", "#F3F4F6"]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.ctaGradient}
          >
            <Text style={styles.ctaText}>
              {isListening ? "Listening..." : "What's on your mind?"}
            </Text>
            <View style={styles.ctaArrow}>
              <Text style={styles.ctaArrowText}>→</Text>
            </View>
          </LinearGradient>
        </Pressable>
      </LinearGradient>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#FFFFFF",
  },
  scrollView: {
    flex: 1,
    backgroundColor: "#FFFFFF",
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 8,
  },
  whiteContent: {
    backgroundColor: "#FFFFFF",
    paddingBottom: 20,
    gap: 12,
  },
  calendarScrollView: {
    marginBottom: 12,
  },
  calendarContainer: {
    paddingHorizontal: 20,
  },
  tabNumber: {
    width: 32,
    height: 32,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
  },
  tabNumberLast: {
    marginRight: 0,
  },
  tabNumberToday: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: "#FF0000",
    justifyContent: "center",
    alignItems: "center",
  },
  tabNumberText: {
    fontSize: 16,
    color: "#000000",
    fontWeight: "500",
  },
  tabNumberTextToday: {
    color: "#FFFFFF",
  },
  transcriptContainer: {
    marginBottom: 24,
    gap: 8,
  },
  transcriptLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#6B7280",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  transcriptCard: {
    backgroundColor: "#F0F9FF",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#BAE6FD",
    padding: 16,
  },
  transcriptText: {
    fontSize: 16,
    color: "#0C4A6E",
    lineHeight: 24,
  },
  ideasContainer: {
    gap: 12,
    marginBottom: 24,
  },
  ideaCard: {
    backgroundColor: "#F5F5F5",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#E0E0E0",
    padding: 16,
  },
  ideaText: {
    fontSize: 16,
    color: "#000000",
  },
  todoHeading: {
    fontSize: 18,
    fontWeight: "600",
    color: "#000000",
    marginBottom: 12,
  },
  todoItem: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 12,
    gap: 12,
  },
  checkbox: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: "#000000",
    backgroundColor: "#FFFFFF",
  },
  todoText: {
    fontSize: 16,
    color: "#000000",
  },
  blackCard: {
    borderRadius: 20,
    padding: 4,
    margin: 12,
    marginBottom: 0,
    alignItems: "center",
    justifyContent: "center",
  },
  ctaButton: {
    width: "100%",
    borderRadius: 16,
    overflow: "hidden",
    shadowColor: "#000000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 8,
  },
  ctaButtonPressed: {
    opacity: 0.9,
    transform: [{ scale: 0.98 }],
  },
  ctaGradient: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 24,
    paddingVertical: 18,
    gap: 12,
  },
  ctaText: {
    fontSize: 18,
    fontWeight: "600",
    color: "#1F2937",
    letterSpacing: 0.3,
  },
  ctaArrow: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: "rgba(31, 41, 55, 0.1)",
    alignItems: "center",
    justifyContent: "center",
  },
  ctaArrowText: {
    fontSize: 16,
    color: "#1F2937",
    fontWeight: "700",
  },
});
