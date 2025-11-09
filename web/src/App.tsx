import { useState, useEffect, useRef } from "react";
import SpeechRecognition, {
  useSpeechRecognition,
} from "react-speech-recognition";
import "./App.css";

interface DateNotes {
  ideas: string[];
  todos: string[];
}

function App() {
  const [dates, setDates] = useState<Date[]>([]);
  const [selectedDate, setSelectedDate] = useState<Date | null>(null);
  const [notesByDate, setNotesByDate] = useState<Map<string, DateNotes>>(
    new Map()
  );
  const [completedTodos, setCompletedTodos] = useState<
    Map<string, Set<number>>
  >(new Map());
  const calendarScrollRef = useRef<HTMLDivElement>(null);

  // Get today's date normalized to start of day
  const getToday = () => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return today;
  };

  // Get date key string for Map storage
  const getDateKey = (date: Date) => {
    return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;
  };

  const { transcript, listening } = useSpeechRecognition();

  // Generate dates: 30 days back from today (including today)
  useEffect(() => {
    const today = getToday();
    const dateArray: Date[] = [];
    const initialNotes = new Map<string, DateNotes>();

    for (let i = -30; i <= 0; i++) {
      const date = new Date(today);
      date.setDate(today.getDate() + i);
      dateArray.push(date);

      // Initialize sample notes for different dates
      const dateKey = getDateKey(date);
      const daysFromToday = Math.abs(i);

      // Different notes for different dates
      if (i === 0) {
        // Today
        initialNotes.set(dateKey, {
          ideas: [
            "Idea to build a speech-based AI notes app called Mindflow",
            "Consider adding voice commands for navigation",
          ],
          todos: [
            "Build Mindflow",
            "Test voice recognition",
            "Add date selection feature",
          ],
        });
      } else if (i === -1) {
        // Yesterday
        initialNotes.set(dateKey, {
          ideas: [
            "Research speech recognition libraries",
            "Design the UI layout for mobile and web",
          ],
          todos: ["Set up project structure", "Install dependencies"],
        });
      } else if (i === -2) {
        // 2 days ago
        initialNotes.set(dateKey, {
          ideas: [
            "Create a mind mapping feature",
            "Add support for multiple languages",
          ],
          todos: ["Plan the architecture", "Create wireframes"],
        });
      } else if (daysFromToday % 7 === 0) {
        // Weekly dates
        initialNotes.set(dateKey, {
          ideas: [
            `Weekly reflection from ${daysFromToday} days ago`,
            "Review progress and plan next steps",
          ],
          todos: ["Weekly review", "Update project timeline"],
        });
      } else {
        // Default notes for other dates
        initialNotes.set(dateKey, {
          ideas: [
            `Notes from ${daysFromToday} days ago`,
            "Keep building and iterating",
          ],
          todos: [
            `Task from ${daysFromToday} days ago`,
            "Continue development",
          ],
        });
      }
    }

    setDates(dateArray);
    setSelectedDate(today);
    setNotesByDate(initialNotes);

    // Scroll to today's date on mount
    setTimeout(() => {
      if (calendarScrollRef.current) {
        const todayIndex = dateArray.length - 1;
        const itemWidth = 48; // Updated to match new fixed width
        const itemGap = 12;
        const scrollPosition = todayIndex * (itemWidth + itemGap) - 20;
        calendarScrollRef.current.scrollLeft = Math.max(0, scrollPosition);
      }
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

  const isSelected = (date: Date) => {
    if (!selectedDate) return false;
    return (
      date.getDate() === selectedDate.getDate() &&
      date.getMonth() === selectedDate.getMonth() &&
      date.getFullYear() === selectedDate.getFullYear()
    );
  };

  const startListening = () => {
    SpeechRecognition.startListening({ continuous: true });
  };

  const stopListening = async () => {
    SpeechRecognition.stopListening();

    if (!transcript.trim() || !selectedDate) return;

    const res = await fetch("http://localhost:8000/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: transcript }),
    });
    const data = await res.json();
    console.log(data);

    // Process the API response and add items to the appropriate lists
    if (Array.isArray(data)) {
      const dateKey = getDateKey(selectedDate);
      setNotesByDate((prev) => {
        const newMap = new Map(prev);
        const currentNotes = newMap.get(dateKey) || { ideas: [], todos: [] };

        // Create new arrays with existing items
        const newIdeas = [...currentNotes.ideas];
        const newTodos = [...currentNotes.todos];

        // Process each item from the API response
        data.forEach((item: { type_item: string; text: string }) => {
          if (item.type_item === "todo") {
            newTodos.push(item.text);
          } else if (item.type_item === "note") {
            newIdeas.push(item.text);
          }
        });

        // Update the map with the new notes
        newMap.set(dateKey, {
          ideas: newIdeas,
          todos: newTodos,
        });

        return newMap;
      });
    }
  };

  const toggleTodo = (dateKey: string, todoIndex: number) => {
    setCompletedTodos((prev) => {
      const newMap = new Map(prev);
      const completedSet = newMap.get(dateKey) || new Set<number>();
      const newSet = new Set(completedSet);

      if (newSet.has(todoIndex)) {
        newSet.delete(todoIndex);
      } else {
        newSet.add(todoIndex);
      }

      newMap.set(dateKey, newSet);
      return newMap;
    });
  };

  const isTodoCompleted = (dateKey: string, todoIndex: number) => {
    return completedTodos.get(dateKey)?.has(todoIndex) || false;
  };

  return (
    <div className="flex flex-col h-screen">
      <div className="min-h-screen max-w-4xl mx-auto bg-white flex flex-col">
        {/* White Content Area */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto px-5 pt-5 pb-8">
            {/* Calendar Scroll */}
            <div
              ref={calendarScrollRef}
              className="flex overflow-x-auto gap-3 mb-3 pb-2 scrollbar-hide"
              style={{
                scrollSnapType: "x proximity",
              }}
            >
              {dates.map((date, index) => {
                const isTodayDate = isToday(date);
                const isSelectedDate = isSelected(date);
                return (
                  <button
                    key={index}
                    onClick={() => setSelectedDate(date)}
                    className={`
                    w-12 h-12 flex items-center justify-center shrink-0
                    rounded-full
                    ${isTodayDate ? "bg-red-500" : ""}
                    ${
                      isSelectedDate && !isTodayDate
                        ? "bg-gray-200"
                        : !isTodayDate
                        ? "bg-white border-2 border-gray-300"
                        : ""
                    }
                    transition-colors hover:opacity-80
                  `}
                    style={{
                      scrollSnapAlign: "start",
                    }}
                  >
                    <span
                      className={`text-base font-medium ${
                        isTodayDate ? "text-white" : "text-black"
                      }`}
                    >
                      {formatDay(date)}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Idea Entries */}
            {selectedDate &&
              (() => {
                const currentNotes = notesByDate.get(getDateKey(selectedDate));
                const ideas = currentNotes?.ideas || [];
                const todos = currentNotes?.todos || [];

                return (
                  <>
                    <div className="space-y-3 mb-6">
                      {ideas.length > 0 ? (
                        ideas.map((idea, index) => (
                          <div
                            key={index}
                            className="bg-gray-100 rounded-xl border border-gray-200 p-4"
                          >
                            <p className="text-base text-black">{idea}</p>
                          </div>
                        ))
                      ) : (
                        <div className="bg-gray-100 rounded-xl border border-gray-200 p-4">
                          <p className="text-base text-gray-500">
                            No ideas for this date
                          </p>
                        </div>
                      )}
                    </div>

                    {/* To-do Section */}
                    <div>
                      <h2 className="text-lg font-semibold text-black mb-3">
                        To-do:
                      </h2>
                      <div className="space-y-3">
                        {todos.length > 0 ? (
                          todos.map((todo, index) => {
                            const dateKey = getDateKey(selectedDate!);
                            const isCompleted = isTodoCompleted(dateKey, index);
                            return (
                              <button
                                key={index}
                                onClick={() => toggleTodo(dateKey, index)}
                                className="flex items-center gap-3 w-full text-left hover:opacity-80 transition-opacity"
                              >
                                <div
                                  className={`w-5 h-5 rounded-full border-2 border-black shrink-0 flex items-center justify-center ${
                                    isCompleted ? "bg-black" : "bg-white"
                                  }`}
                                >
                                  {isCompleted && (
                                    <span className="text-white text-xs font-bold">
                                      ✓
                                    </span>
                                  )}
                                </div>
                                <span
                                  className={`text-base ${
                                    isCompleted
                                      ? "text-gray-500 line-through"
                                      : "text-black"
                                  }`}
                                >
                                  {todo}
                                </span>
                              </button>
                            );
                          })
                        ) : (
                          <div className="flex items-center gap-3">
                            <div className="w-5 h-5 rounded-full border-2 border-black bg-white shrink-0" />
                            <span className="text-base text-gray-500">
                              No todos for this date
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                );
              })()}
          </div>
        </div>
        <div className="bg-gray-100 rounded-xl border text-center  border-gray-200 p-4">
          <p>{transcript}</p>
        </div>

        {/* Bottom Gradient Card */}
        <div className="bg-gradient-to-r from-purple-600 via-pink-500 to-red-500 rounded-2xl p-1 mx-3 mb-3">
          <button
            onClick={listening ? stopListening : startListening}
            className="w-full bg-gradient-to-r from-white to-gray-100 rounded-2xl px-6 py-4 flex items-center justify-center gap-3 shadow-lg hover:opacity-90 transition-opacity"
          >
            <span className="flex-1 text-lg font-semibold text-gray-800 text-left">
              {listening ? "Listening..." : "What's on your mind?"}
            </span>
            <div className="w-6 h-6 rounded-full bg-gray-200 flex items-center justify-center">
              <span className="text-base font-bold text-gray-800">→</span>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
